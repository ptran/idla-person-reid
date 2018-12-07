#include "dataset.h"
#include <algorithm>
#include <fstream>
#include <dlib/dir_nav.h>
#include <dlib/image_io.h>
#include <dlib/string.h>

void load_cuhk03_dataset(
    std::string dirpath,
    std::vector<two_view_images>& person_images,
    std::vector<std::vector<std::size_t>>& testsets,
    cuhk03_dataset_type dataset_type,
    long nr,
    long nc
)
{
    person_images.resize(1467);
    // Load images into memory
    dlib::directory image_dir(dirpath+"/"+((dataset_type == LABELED) ? "labeled" : "detected"));
    auto image_files = image_dir.get_files();
    for (auto& f : image_files) {
        // Extract person and view index
        std::string image = f.name();
        std::size_t pid = std::stoul(image.substr(0, 4));
        std::size_t vid = std::stoul(image.substr(5, 2));

        // Load image
        dlib::array2d<dlib::rgb_pixel> img;
        dlib::load_image(img, f.full_name());

        // Resize based on input
        dlib::array2d<dlib::rgb_pixel> resized(nr, nc);
        dlib::resize_image(img, resized);

        auto& view_imgs = (vid < 5) ? person_images[pid].first : person_images[pid].second;
        view_imgs.push_back(dlib::mat(resized));
    }

    // Load test indices 
    testsets.resize(20);
    std::string test_path = dirpath + "/testsets.csv";

    std::ifstream ifs;
    ifs.open(test_path);
    for (int i = 0; i < 20; ++i) {
        testsets[i].resize(100);
        std::string line;
        std::getline(ifs, line);
        line = dlib::split_on_last(line, "\n").first;

        auto tokens = dlib::split(line, ",");
        for (std::size_t t = 0; t < tokens.size(); ++t) {
            testsets[i][t] = std::stoul(tokens[t]);
        }
    }
}
