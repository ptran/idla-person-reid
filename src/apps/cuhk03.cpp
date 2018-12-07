#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/rand.h>

#include "dataset.h"
#include "difference.h"
#include "input.h"
#include "multiclass_less.h"
#include "reinterpret.h"

// Neural Network Setup
// ---------------------------------------------------------------------------

template <
    long num_filters,
    long nr,
    long nc,
    int stride_y,
    int stride_x,
    typename SUBNET
    >
using connp = dlib::add_layer<dlib::con_<num_filters,nr,nc,stride_y,stride_x,0,0>, SUBNET>;

template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using block = dlib::relu<BN<connp<N, shape, shape, stride, stride, SUBNET>>>;

template <template <typename> class BN_CON, template <typename> class BN_FC>
using mod_idla = loss_multiclass_log_lr<dlib::fc<2,
                                        dlib::relu<BN_FC<dlib::fc<500,reinterpret<2,
                                        dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                        block<25,BN_CON,5,5, // patch summary
                                        dlib::relu<cross_neighborhood_differences<5,5,
                                        dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,block<25,BN_CON,3,1,
                                        dlib::max_pool<2,2,2,2,block<20,BN_CON,3,1,block<20,BN_CON,3,1,
                                        input_rgb_image_pair
                                        >>>>>>>>>>>>>>>>>;

using net_type = mod_idla<dlib::bn_con, dlib::bn_fc>;    // Training Net
using anet_type = mod_idla<dlib::affine, dlib::affine>;  // Testing Net

// Minibatch Generation
// ---------------------------------------------------------------------------
typedef input_rgb_image_pair::input_type input_type;

struct minibatch {
    std::vector<input_type> image_pairs;
    std::vector<unsigned long> labels;
};

class minibatch_generator {
public:
    minibatch_generator(const std::vector<two_view_images>& images_, const std::vector<std::size_t>& test_indices)
        : images(images_)
    {
        for (std::size_t i = 0; i < this->images.size(); ++i) {
            if (std::find(test_indices.begin(), test_indices.end(), i) == test_indices.end())
                train_indices.push_back(i);
        }
    }

    minibatch operator()(std::size_t size)
    {
        DLIB_CASSERT(size % 2 == 0, "");

        // Sample training indices
        dlib::random_subset_selector<std::size_t> samples;
        while (true) {
            unsigned int seed = rng.get_random_32bit_number();
            samples = dlib::randomly_subsample(train_indices, size, seed);
            std::size_t i;
            for (i = 0; i < size/2; ++i) {
                // Re-sample if the positive image pair and negative gallery image
                std::size_t p = samples[i];
                std::size_t n = samples[i+size/2];
                if (images[p].first.empty() || images[p].second.empty() || images[n].second.empty())
                    break;
            }
            if (i >= size/2)
                break;
        }

        // Build minibatch
        std::vector<std::pair<input_type, unsigned long>> batch_pairs;
        for (unsigned long i = 0; i < size/2; ++i) {
            std::size_t p = samples[i];
            std::size_t n = samples[i+size/2];

            auto& probe = images[p].first;
            auto& gallery_p = images[p].second;

            std::size_t ppi = rng.get_random_32bit_number() % probe.size();
            std::size_t gpi = rng.get_random_32bit_number() % gallery_p.size();
            batch_pairs.emplace_back(std::make_pair(&probe[ppi], &gallery_p[gpi]), 1);

            auto& gallery_n = images[n].second;
            unsigned int pni = rng.get_random_32bit_number() % probe.size();
            unsigned int gni = rng.get_random_32bit_number() % gallery_n.size();
            batch_pairs.emplace_back(std::make_pair(&probe[pni], &gallery_n[gni]), 0);
        }
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(batch_pairs), std::end(batch_pairs), engine);

        minibatch mb;
        mb.image_pairs.reserve(size);
        mb.labels.reserve(size);
        for (auto bp : batch_pairs) {
            mb.image_pairs.push_back(std::move(bp.first));
            mb.labels.push_back(std::move(bp.second));
        }
        return mb;
    }
private:
    dlib::rand rng;
    const std::vector<two_view_images>& images;
    std::vector<std::size_t> train_indices;
};

int main(int argc, char* argv[]) try
{
    dlib::command_line_parser parser;
    parser.add_option("i", "Directory holding the CUHK03 dataset", 1);
    parser.add_option("detected", "Indicates the 'detected' dataset should be used. 'labeled' is used by default.");
    parser.add_option("h", "Display a help message.");

    // Parse command line arguments
    parser.parse(argc, argv);
    if (parser.option("h")) {
        std::cout << "Usage: run_cuhk03 [--detected] -i cuhk03_dir\n";
        parser.print_options();
        return 0;
    }

    if (!parser.option("i")) {
        std::cout << "You must specify the i option (input directory).\n";
        std::cout << "\n Try the -h option for more information." << std::endl;
        return 0;
    }

    // Load in dataset and time it
    std::string cuhk03_dir = parser.option("i").argument();
    cuhk03_dataset_type dataset_type = parser.option("detected") ? DETECTED : LABELED;
    std::cout << "Attempting to load the CUHK03 "
              << ((dataset_type == LABELED) ? "labeled" : "detected")
              << " dataset from '" << cuhk03_dir << "' [should take up to 5 seconds in release mode]..."
              << std::endl;

    std::vector<two_view_images> person_images;
    std::vector<std::vector<std::size_t>> testsets;

    auto start = std::chrono::system_clock::now();
    load_cuhk03_dataset(cuhk03_dir, person_images, testsets, dataset_type);
    auto stop = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop-start;
    std::cout << elapsed_seconds.count() << " seconds to load dataset." << std::endl;

    // Start training code
    net_type net;
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.be_verbose();

    // Set learning rate schedule
    unsigned long max_iterations = 210000;
    unsigned long current_iteration = trainer.get_train_one_step_calls();

    dlib::matrix<double,0,1> inverse_learning_rate_schedule;
    inverse_learning_rate_schedule.set_size(max_iterations-current_iteration);

    double learning_rate = 0.01;
    trainer.set_learning_rate(learning_rate);

    double gamma = 0.0001;
    double power = 0.75;
    for (unsigned long i = current_iteration; i < max_iterations; ++i) {
        inverse_learning_rate_schedule(i-current_iteration) = learning_rate*std::pow(1.0+gamma*i, -power);
    }
    trainer.set_learning_rate_schedule(inverse_learning_rate_schedule);

    // Save training progress
    std::string save_name;
    {
        std::ostringstream oss;
        oss << "cuhk03_" << ((dataset_type == LABELED) ? "labeled" : "detected") << "_modidla";
        save_name = oss.str();
    }
    trainer.set_synchronization_file(save_name+".dat", std::chrono::seconds(60));

    // Prepare data
    long batch_size = 128;
    dlib::rand rng(0);
    auto& test_indices = testsets[rng.get_random_32bit_number() % 20];
    minibatch_generator batchgen(person_images, test_indices);

    // Train neural network
    std::cout << std::endl
              << net
              << std::endl;
    while (trainer.get_train_one_step_calls() < max_iterations) {
        minibatch mb = batchgen(batch_size);
        trainer.train_one_step(mb.image_pairs.begin(), mb.image_pairs.end(), mb.labels.begin());
    }
    trainer.get_net();

    // Save the network to disk
    net.clean();
    std::cout << "Saving network..." << std::endl;
    dlib::serialize(save_name+".dnn") << net;

    // Test the network on the CUHK03 testing data.
    dlib::softmax<anet_type::subnet_type> tnet;
    tnet.subnet() = net.subnet();
    std::cout << "Testing network on CUHK03 testing dataset." << std::endl;

    // Use the specified test indices for evaluation
    std::vector<int> ranked_counter(test_indices.size(), 0);
    int num_probes = 0;

    const int num_trials = 100;
    dlib::console_progress_indicator pbar(test_indices.size());
    int pctr = 0;
    for (auto pid : test_indices) {
        pbar.print_status(pctr++);

        const auto& probe_imgs = person_images[pid].first;
        for (const auto& probe_img : probe_imgs) {
            // Count total number of probe images
            ++num_probes;
            std::vector<std::vector<std::pair<float, std::size_t>>> trials(num_trials);
            for (auto gid : test_indices) {
                const auto& gallery_imgs = person_images[gid].second;
                std::vector<input_type> img_pairs;
                img_pairs.reserve(gallery_imgs.size());
                for (const auto& gallery_img : gallery_imgs) {
                    img_pairs.emplace_back(&probe_img, &gallery_img);
                }

                // Randomly choose one pairwise score to represent the current gallery person ID for each trial
                dlib::matrix<float> output = dlib::mat(tnet(img_pairs.begin(), img_pairs.end()));
                for (auto& trial : trials) {
                    int rnd_idx = rng.get_random_32bit_number() % output.nr();
                    float score = output(rnd_idx, 1);
                    trial.emplace_back(score, gid);
                }
            }
            // Each trial contains score and gallery ID pairs
            for (auto& trial : trials) {
                // Sort score and ID pairs and scan for the matching ID
                std::sort(trial.begin(), trial.end(),
                          [](const std::pair<float, int>& a, const std::pair<float, int>& b) -> bool
                          {
                              return a.first > b.first;
                          });

                // Find the first occurrence of the same ID person
                for (std::size_t i = 0; i < trial.size(); ++i) {
                    auto& score_pair = trial[i];
                    auto gid = score_pair.second;
                    if (pid == gid) {
                        ++ranked_counter[i];
                        break;
                    }
                }
            }
        }
    }

    // Calculate the cumulative match curve for this dataset.
    dlib::matrix<double> cmc;
    cmc.set_size(1, ranked_counter.size());
    int accumulated_count = 0;

    std::ofstream cmc_file;
    cmc_file.open("cmc_"+save_name+".csv");
    for (std::size_t i = 0; i < ranked_counter.size(); ++i) {
        accumulated_count += ranked_counter[i];
        cmc(i) = static_cast<double>(accumulated_count)/(num_probes*num_trials);
        cmc_file << cmc(i) << ((i < (ranked_counter.size()-1)) ? "," : "\n");
    }
    std::cout << "\nCumulative match curve saved to `cmc_cuhk03_modidla.csv`." << std::endl;

    return 0;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
