#ifndef IDLA__DATASET_H_
#define IDLA__DATASET_H_

#include <string>
#include <utility>
#include <dlib/array.h>
#include <dlib/image_transforms.h>

typedef dlib::array2d<dlib::rgb_pixel> rgb_image;
typedef std::pair<dlib::array<rgb_image>, dlib::array<rgb_image>> two_view_images;

enum cuhk03_dataset_type {
    LABELED,
    DETECTED
};

/*!
    Loads the CUHK03 dataset.

    ensures:
        - images has 1467 entries, each with images for a particular person
        - testsets will have 20 entries, each with 100 indices that should
          be used for as test data.

    throws:
        - std::runtime_error, if given file is not the expected CUHK03 mat file.
*/
void load_cuhk03_dataset(
    std::string dirpath,
    dlib::array<two_view_images>& person_images,
    std::vector<std::vector<std::size_t>>& testsets,
    cuhk03_dataset_type dataset_type=LABELED
);

#endif // IDLA__DATASET_H_
