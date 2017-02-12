#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>

#include "dataset.h"
#include "difference.h"
#include "input.h"
#include "reinterpret.h"

// Define the network type
template <long N, template <typename> class BN, long shape, long stride, typename SUBNET>
using block = dlib::relu<BN<dlib::con<N, shape, shape, stride, stride, SUBNET>>>;

template <long HN, template <typename> class BN_CON, template <typename> class BN_FC>
using mod_idla = dlib::loss_multiclass_log<dlib::fc<10,
                                           BN_FC<dlib::fc<500,reinterpret<2,
                                           dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                           block<25,BN_CON,5,5, // Patch Summary Features
                                           cross_neighborhood_differences<5,5,
                                           dlib::max_pool<2,2,2,2,block<25,BN_CON,3,1,
                                           block<25,BN_CON,3,1,
                                           dlib::max_pool<2,2,2,2,block<20,BN_CON,3,1,
                                           block<20,BN_CON,3,1,
                                           input_rgb_image_pair>
                                           >>>>>>>>>>>>>>;

using net_type = mod_idla<64, dlib::bn_con, dlib::bn_fc>;
using anet_type = mod_idla<64, dlib::affine, dlib::affine>;

int main(int argc, char* argv[]) try
{
    dlib::command_line_parser parser;
    parser.add_option("i", "Directory holding the CUHK03 dataset", 1);
    parser.add_option("detected", "Indicates the 'detected' dataset should be used. 'labeled' is used by default.");
    parser.add_option("h", "Display a help message.");

    // Parse command line arguments
    parser.parse(argc, argv);
    if (parser.option("h")) {
        std::cout << "Usage: run_reid [--detected] -i cuhk03_dir\n";
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

    return 0;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
}
