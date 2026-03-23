#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/console_progress_indicator.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/rand.h>

#include "network.h"
#include "dataset.h"

// ---------------------------------------------------------------------------

typedef input_rgb_image_pair::input_type input_type;

struct minibatch {
    std::vector<input_type> data;
    std::vector<unsigned long> labels;
};

class minibatch_generator {
public:
    minibatch_generator(
        const std::vector<person_set>& pset_,
        const std::vector<int>& tidx
    ) : pset(pset_)
    {
        for (unsigned long i = 0; i < pset_.size(); ++i) {
            if (std::find(tidx.begin(), tidx.end(), i) == tidx.end())
                tridx.push_back(i);
        }
    }

    minibatch operator()(unsigned long size)
    {
        DLIB_CASSERT(size % 2 == 0, "");

        // Create random sampling object
        dlib::random_subset_selector<int> samples;
        bool empty_view = true;
        while (empty_view) {
            unsigned int seed = rng.get_random_32bit_number();
            samples = dlib::randomly_subsample(tridx, size, seed);

            empty_view = false;
            for (unsigned int i = 0; i < size/2; ++i) {
                const auto& v0 = pset[samples[i]].view(0);
                const auto& v1p = pset[samples[i]].view(1);
                const auto& v1n = pset[samples[i+size/2]].view(1);
                if (v0.size() == 0 || v1p.size() == 0 || v1n.size() == 0) {
                    empty_view = true;
                    break;
                }
            }
        }

        minibatch batch;
        batch.data.reserve(size);
        batch.labels.reserve(size);

        for (unsigned int i = 0; i < size/2; ++i) {
            // Pick random indices
            const auto& view0 = pset[samples[i]].view(0);
            const auto& view1p = pset[samples[i]].view(1);
            const auto& view1n = pset[samples[i+size/2]].view(1);

            unsigned int pidx0 = rng.get_random_32bit_number() % view0.size();
            unsigned int pidx1 = rng.get_random_32bit_number() % view1p.size();
            unsigned int nidx1 = rng.get_random_32bit_number() % view1n.size();

            // Positive pair
            batch.data.push_back({&view0[pidx0], &view1p[pidx1]});
            batch.labels.push_back(1);

            // Negative pair (uses same view0)
            batch.data.push_back({&view0[pidx0], &view1n[nidx1]});
            batch.labels.push_back(0);
        }

        return batch;
    }
private:
    dlib::rand rng;
    const std::vector<person_set>& pset; 
    std::vector<int> tridx;                //  training index
};

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) try
{
    // Memory-saving flag recommended by dlib documentation for OOM issues
    dlib::set_dnn_prefer_smallest_algorithms();

    // --- EAGER MEMORY ALLOCATION ---
    // Instantiate network and trainer FIRST to secure contiguous pinned memory 
    // before the heap is fragmented by thousands of small dataset matrices.
    net_type net;
    dlib::dnn_trainer<net_type, dlib::adam> trainer(net, dlib::adam(0.0005, 0.9, 0.999));
    trainer.set_mini_batch_size(32); 
    trainer.be_verbose();
    trainer.set_learning_rate(0.001);

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
        std::cout << "Error: You must provide the CUHK03 dataset directory with the -i option.\n";
        return 1;
    }

    const std::string cuhk03_dir = parser.option("i").argument();
    const std::string cuhk03_file = cuhk03_dir + "/cuhk-03.mat";

    cuhk03_dataset_type dset_type = parser.option("detected") ? DETECTED : LABELED;
    std::cout << "Attempting to load the CUHK03 " << ((dset_type == LABELED) ? "labeled" : "detected")
              << " dataset from '" << cuhk03_dir << "/'..." << std::endl;

    // Load dataset references
    std::vector<person_set> pset;
    std::vector<std::vector<int>> test_protocols;
    load_cuhk03_dataset(cuhk03_file, pset, test_protocols, dset_type);

    // Set learning rate schedule
    const unsigned long max_iterations = 50000;
    const unsigned long current_iteration = trainer.get_train_one_step_calls();

    dlib::matrix<double,0,1> inverse_learning_rate_schedule;
    inverse_learning_rate_schedule.set_size(max_iterations-current_iteration);

    double learning_rate = 0.001;
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
        oss << "cuhk03_" << ((dset_type == LABELED) ? "labeled" : "detected") << "_modidla";
        save_name = oss.str();
    }
    trainer.set_synchronization_file(save_name+".dat", std::chrono::seconds(60));

    // Prepare data
    long batch_size = 32;
    dlib::rand rng(0);
    unsigned int test_index = rng.get_random_32bit_number() % 20;
    minibatch_generator batchgen(pset, test_protocols[test_index]);

    // Train neural network
    std::cout << std::endl << net << std::endl;
    while (trainer.get_train_one_step_calls() < max_iterations) {
        minibatch batch = batchgen(batch_size);
        trainer.train_one_step(batch.data.begin(), batch.data.end(), batch.labels.begin());
    }
    trainer.get_net();

    // Save the network to disk
    net.clean();
    std::cout << "Saving network..." << std::endl;
    dlib::serialize(save_name+".dnn") << net;

    // Evaluation
    dlib::softmax<anet_type::subnet_type> tnet;
    tnet.subnet() = net.subnet();
    std::cout << "Testing network on CUHK03 testing dataset." << std::endl;

    std::vector<int> ranked_counter(test_protocols[test_index].size(), 0);
    int num_probes = 0;

    const int num_trials = 100;
    dlib::console_progress_indicator pbar(test_protocols[test_index].size());
    for (unsigned int i = 0; i < test_protocols[test_index].size(); ++i) {
        int pid = test_protocols[test_index][i];
        pbar.print_status(i);
        for (unsigned int v0_idx = 0; v0_idx < pset[pid].view(0).size(); ++v0_idx) {
            ++num_probes;
            const dlib::matrix<dlib::rgb_pixel>& probe_img = pset[pid].view(0)[v0_idx];

            std::vector<std::vector<std::pair<float,int>>> trials(num_trials);
            for (int t = 0; t < num_trials; ++t) trials[t].reserve(test_protocols[test_index].size());

            for (unsigned int j = 0; j < test_protocols[test_index].size(); ++j) {
                int gid = test_protocols[test_index][j];
                std::vector<input_type> img_pairs;
                for (unsigned int v1_idx = 0; v1_idx < pset[gid].view(1).size(); ++v1_idx) {
                    img_pairs.push_back({&probe_img, &pset[gid].view(1)[v1_idx]});
                }

                dlib::matrix<float> output = dlib::mat(tnet(img_pairs.begin(), img_pairs.end()));
                for (auto& trial : trials) {
                    int tmp = rng.get_random_32bit_number() % output.nr();
                    trial.push_back(std::make_pair(output(tmp, 1), gid));
                }
            }

            for (auto& trial : trials) {
                std::sort(trial.begin(), trial.end(), [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                    return a.first > b.first;
                });
                for (unsigned int j = 0; j < trial.size(); ++j) {
                    if (pid == trial[j].second) {
                        ++ranked_counter[j];
                        break;
                    }
                }
            }
        }
    }

    std::ofstream cmc_file("cmc_"+save_name+".csv");
    int accumulated_count = 0;
    for (unsigned int i = 0; i < ranked_counter.size(); ++i) {
        accumulated_count += ranked_counter[i];
        double cmc_val = (double)accumulated_count/(num_probes*num_trials);
        cmc_file << cmc_val << ((i < (ranked_counter.size()-1)) ? "," : "\n");
    }

    return 0;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
