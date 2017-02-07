#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/gui_widgets.h>

#include "dataset.h"
#include "difference.h"

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
    std::cout << "Attempting to load the CUHK03 dataset from '" << cuhk03_dir << "'..." << std::endl;
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
