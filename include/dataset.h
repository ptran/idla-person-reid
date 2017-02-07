#ifndef IDLA__DATASET_H_
#define IDLA__DATASET_H_

#include <string>
#include <utility>

#include <dlib/noncopyable.h>
#include <dlib/image_transforms.h>

// ---------------------------------------------------------------------------

/*!
    Container of images of a single person across multiple cameras/views.
*/
class person_set : dlib::noncopyable {
public:
    person_set(std::vector<std::vector<dlib::matrix<dlib::rgb_pixel>>>& images);
    person_set(person_set&& other);

    /*!
        ensures:
            - returns the number of views for this particular person.
    */
    unsigned long get_num_views() const;

    /*!
        requires:
            - view_index < #get_num_views()

        ensures:
            - returns the images from the input view index.
    */
    std::vector<dlib::matrix<dlib::rgb_pixel>>& view(unsigned int view_index);
    const std::vector<dlib::matrix<dlib::rgb_pixel>>& view(unsigned int view_index) const;
private:
    std::vector<std::vector<dlib::matrix<dlib::rgb_pixel>>> images;
};

// ---------------------------------------------------------------------------

enum cuhk03_dataset_type {
    LABELED, DETECTED
};

/*!
    Loads the CUHK03 dataset from the given file.

    requires:
        - dlib::file_exists(cuhk03_file)
        - nr > 0 && nc > 0

    ensures:
        - images has 1467 entries, each with images for a particular person. 
          Each image is scaled to a resolution of nr by nc pixels.
        - test_protocols will have 20 entries, each with 100 indices that should
          be used for as test data.

    throws:
        - std::runtime_error, if given file is not the expected CUHK03 mat file.
*/
void load_cuhk03_dataset(
    const std::string& cuhk03_file,
    std::vector<person_set>& images,
    std::vector<std::vector<int>> test_protocols,
    cuhk03_dataset_type type=LABELED,
    long nr=160,
    long nc=60
);

// ---------------------------------------------------------------------------

#endif // IDLA__DATASET_H_
