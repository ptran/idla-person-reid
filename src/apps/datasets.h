#ifndef IDLA__DATASET_H_
#define IDLA__DATASET_H_

#include <string>
#include <utility>
#include <vector>
#include <dlib/image_transforms.h>

typedef dlib::matrix<dlib::rgb_pixel> rgb_image;
typedef std::pair<std::vector<rgb_image>, std::vector<rgb_image>> two_view_images;

enum cuhk03_dataset_type {
    LABELED,
    DETECTED
};

/*!
    Loads the CUHK03 dataset.

    requires:
        - nr > 0 && nc > 0

    ensures:
        - images has 1467 entries, each with images for a particular person. 
          Each image is scaled to a resolution of nr by nc pixels.
        - testsets will have 20 entries, each with 100 indices that should
          be used for as test data.

    throws:
        - std::runtime_error, if given file is not the expected CUHK03 mat file.
*/
void load_cuhk03_dataset(
    std::string dirpath,
    std::vector<two_view_images>& person_images,
    std::vector<std::vector<std::size_t>>& testsets,
    cuhk03_dataset_type dataset_type=LABELED,
    long nr=160,
    long nc=60
);

#endif // IDLA__DATASET_H_
