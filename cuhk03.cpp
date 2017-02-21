#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/rand.h>

#include "dataset.h"
#include "difference.h"
#include "input.h"
#include "multiclass_less.h"
#include "reinterpret.h"

// ---------------------------------------------------------------------------

template <
    long num_filters,
    long nr,
    long nc,
    int stride_y,
    int stride_x,
    typename SUBNET
    >
using con = dlib::add_layer<dlib::con_<num_filters,nr,nc,stride_y,stride_x,0,0>, SUBNET>;

template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using block = dlib::relu<BN<con<N, shape, shape, stride, stride, SUBNET>>>;

template <template <typename> class BN_CON, template <typename> class BN_FC>
using mod_idla = loss_multiclass_log_lr<dlib::fc<2,
                                        dlib::relu<BN_FC<dlib::fc<500,reinterpret<2,
                                        dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                        block<25,BN_CON,5,5, // patch summary
                                        dlib::relu<cross_neighborhood_differences<5,5,
                                        dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,block<25,BN_CON,3,1,
                                        dlib::max_pool<2,2,2,2,block<20,BN_CON,3,1,block<20,BN_CON,3,1,
                                        input_rgb_image_pair>
                                        >>>>>>>>>>>>>>>>;

using net_type = mod_idla<dlib::bn_con, dlib::bn_fc>;    // Training Net
using anet_type = mod_idla<dlib::affine, dlib::affine>;  // Testing Net

// ---------------------------------------------------------------------------

typedef input_rgb_image_pair::input_type input_type;

class minibatch_generator {
public:
    minibatch_generator(
        const std::vector<person_set>& pset_,
        const std::vector<int>& tidx_
    ) : pset(pset_), tidx(tidx_)
    {
        for (unsigned long i = 0; i < pset_.size(); ++i) {
            if (std::find(tidx.begin(), tidx.end(), i) == tidx.end())
                tridx.push_back(i);
        }
    }

    std::vector<input_type> operator()(unsigned long size, bool test=false)
    {
        DLIB_CASSERT(size % 2 == 0, "");

        // Create random sampling object
        unsigned int seed = rng.get_random_32bit_number();
        dlib::random_subset_selector<int> sampler;
        if (!test)
            sampler = dlib::randomly_subsample(tridx, size, seed);
        else
            sampler = dlib::randomly_subsample(tidx, size, seed);

        // Build minibatch
        std::vector<std::pair<input_type,unsigned long>> labeled_minibatch;
        labeled_minibatch.reserve(size);
        for (unsigned long i = 0; i < size/2; ++i) {
            const std::vector<dlib::matrix<dlib::rgb_pixel>>& view0 = pset[sampler[i]].view(0);
            const std::vector<dlib::matrix<dlib::rgb_pixel>>& pview1 = pset[sampler[i]].view(1);

            // Construct positive pair
            const dlib::matrix<dlib::rgb_pixel>& pimg0 = view0[rng.get_random_32bit_number() % view0.size()];
            const dlib::matrix<dlib::rgb_pixel>& pimg1 = pview1[rng.get_random_32bit_number() % pview1.size()];
            input_type ppair = {&pimg0, &pimg1};
            labeled_minibatch.emplace_back(ppair, 1);

            // Construct negative pair
            const std::vector<dlib::matrix<dlib::rgb_pixel>>& nview1 = pset[sampler[i+size/2]].view(1);
            const dlib::matrix<dlib::rgb_pixel>& nimg0 = view0[rng.get_random_32bit_number() % view0.size()];
            const dlib::matrix<dlib::rgb_pixel>& nimg1 = nview1[rng.get_random_32bit_number() % nview1.size()];
            input_type npair = {&nimg0, &nimg1};
            labeled_minibatch.emplace_back(npair, 0);
        }
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(labeled_minibatch), std::end(labeled_minibatch), engine);

        std::vector<input_type> minibatch;
        minibatch.reserve(size);
        labels.clear();
        labels.reserve(size);
        for (auto i : labeled_minibatch) {
            minibatch.push_back(i.first);
            labels.push_back(i.second);
        }
        return minibatch;
    }

    const std::vector<unsigned long>& get_labels() const
    {
        return labels;
    }
private:
    dlib::rand rng;
    const std::vector<person_set>& pset; 
    std::vector<int> tridx;                //  training index
    const std::vector<int>& tidx;          //  testing index
    std::vector<unsigned long> labels;     //  cached labels
};

// ---------------------------------------------------------------------------

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
#if defined _WIN32
    char os_delim = '\\';
#else
    char os_delim = '/';
#endif
    if (cuhk03_dir.back() != os_delim) {
        cuhk03_dir += os_delim;
    }

    std::cout << "Attempting to load the CUHK03 dataset from '" << cuhk03_dir
              << "' [should take up to 15 seconds]..." << std::endl;
    if (!dlib::file_exists(cuhk03_dir+"cuhk-03.mat")) {
        throw std::runtime_error("'"+cuhk03_dir+"' does not contain cuhk-03.mat.");
    }

    // CUHK03 dataset
    std::vector<person_set> pset;
    std::vector<std::vector<int>> test_protocols;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    load_cuhk03_dataset(cuhk03_dir+"cuhk-03.mat", pset, test_protocols);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << elapsed_seconds.count() << " seconds to load dataset." << std::endl;

    // Start training code
    net_type net;
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.be_verbose();

    // Set learning rate schedule
    unsigned long max_iterations = 50000;
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
    trainer.set_synchronization_file("cuhk03_modidla", std::chrono::seconds(20));

    // Prepare data
    long batch_size = 64;
    dlib::rand rng;
    unsigned int test_index = rng.get_random_32bit_number() % 20;
    minibatch_generator batchgen(pset, test_protocols[test_index]);

    while (trainer.get_train_one_step_calls() < max_iterations) {
        std::vector<input_type> minibatch = batchgen(batch_size);
        std::vector<unsigned long> labels = batchgen.get_labels();
        trainer.train_one_step(minibatch.begin(), minibatch.end(), labels.begin());
    }

    // Save the network to disk
    net.clean();
    dlib::serialize("cuhk03_modidla.dat") << net;

    return 0;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
