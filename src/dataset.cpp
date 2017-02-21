#include "dataset.h"

#include <dlib/rand.h>
#include <dlib/matrix.h>
#include <dlib/image_transforms.h>

#include <H5Cpp.h>

namespace
{
    dlib::rand rng;
}

// ---------------------------------------------------------------------------

person_set::person_set(std::vector<std::vector<dlib::matrix<dlib::rgb_pixel>>>& images_)
{
    images.swap(images_);
}

person_set::person_set(person_set&& other)
{
    images.swap(other.images);
}

unsigned long person_set::get_num_views() const
{
    return images.size();
}

std::vector<dlib::matrix<dlib::rgb_pixel>>& person_set::view(unsigned int view_index)
{
    DLIB_CASSERT(view_index < images.size(), "");
    return images[view_index];
}

const std::vector<dlib::matrix<dlib::rgb_pixel>>& person_set::view(unsigned int view_index) const
{
    DLIB_CASSERT(view_index < images.size(), "");
    return images[view_index];
}

// ---------------------------------------------------------------------------

void load_cuhk03_dataset(
    const std::string& cuhk03_file,
    std::vector<person_set>& images,
    std::vector<std::vector<int>>& test_protocols,
    cuhk03_dataset_type type,
    long nr,
    long nc
)
{
    if (!H5::H5File::isHdf5(cuhk03_file.c_str())) {
        throw std::runtime_error(cuhk03_file + " is not an HDF5 file.");
    }

    H5::H5File file_(cuhk03_file.c_str(), H5F_ACC_RDONLY);
    hobj_ref_t ref_objs[5] = {0};
    if (type == LABELED)
        file_.openDataSet("labeled").read(ref_objs, H5::PredType::STD_REF_OBJ);
    else
        file_.openDataSet("detected").read(ref_objs, H5::PredType::STD_REF_OBJ);

    // Go through each object reference and retrieve image data from them.
    std::vector<hsize_t> ref_sizes(5, 0); // store for mapping test indices
    for (unsigned int i = 0; i < 5; ++i) {
        H5::DataSet img_ref_dset(file_, &ref_objs[i]);
        H5::DataSpace img_ref_space = img_ref_dset.getSpace();

        hsize_t dims[2] = {0};
        img_ref_space.getSimpleExtentDims(dims, NULL);
        ref_sizes[i] = dims[1];

        // Each object reference refers to a matrix of image object references.
        // Pull these references and load the underlying images from them.
        dlib::matrix<hobj_ref_t> img_ref_objs(dims[0], dims[1]);
        img_ref_objs = 0;
        img_ref_dset.read(img_ref_objs.begin(), H5::PredType::STD_REF_OBJ);

        for (long c = 0; c < img_ref_objs.nc(); ++c) {
            std::vector<dlib::matrix<dlib::rgb_pixel>> vimgs0, vimgs1;
            vimgs0.reserve(img_ref_objs.nr());
            vimgs1.reserve(img_ref_objs.nr());

            for (long r = 0; r < img_ref_objs.nr(); ++r) {                
                // Get the image dataset
                H5::DataSet img_dset(file_, &img_ref_objs(r, c));
                H5::DataSpace img_space = img_dset.getSpace();
                std::vector<hsize_t> img_dims(img_space.getSimpleExtentNdims());
                if (img_dims.size() != 3)
                    continue;
                img_space.getSimpleExtentDims(&img_dims[0], NULL);

                // Transfer image from HDF5 to memory
                std::vector<unsigned char> img_array(img_dims[0]*img_dims[1]*img_dims[2], 0);
                img_dset.read(&img_array[0], H5::PredType::NATIVE_UINT8);

                dlib::matrix<dlib::rgb_pixel> img_copy(img_dims[2], img_dims[1]);
                unsigned int img_size = img_dims[1]*img_dims[2];
                for (unsigned int k = 0; k < img_size; ++k) {
                    unsigned int r_ = k % img_dims[2];
                    unsigned int c_ = k/img_dims[2] % img_dims[1];

                    img_copy(r_, c_).red = img_array[k];
                    img_copy(r_, c_).green = img_array[k+img_size];
                    img_copy(r_, c_).blue = img_array[k+2*img_size];
                }

                // Resize to final image
                if (r < 5) {
                    vimgs0.emplace_back(nr, nc);
                    dlib::matrix<dlib::rgb_pixel>& img = vimgs0.back();
                    dlib::resize_image(img_copy, img);
                }
                else {
                    vimgs1.emplace_back(nr, nc);
                    dlib::matrix<dlib::rgb_pixel>& img = vimgs1.back();
                    dlib::resize_image(img_copy, img);
                }
            }
            std::vector<std::vector<dlib::matrix<dlib::rgb_pixel>>> pimgs;
            pimgs.push_back(std::move(vimgs0));
            pimgs.push_back(std::move(vimgs1));
            images.emplace_back(pimgs);
        }
    }

    // Load test protocols
    hobj_ref_t test_ref_objs[20] = {0};
    file_.openDataSet("testsets").read(test_ref_objs, H5::PredType::STD_REF_OBJ);
    for (unsigned int i = 0; i < 20; ++i) {
        H5::DataSet test_ref_dset(file_, &test_ref_objs[i]);
        double h5_indices[2][100] = {0.0};
        test_ref_dset.read(h5_indices, H5::PredType::NATIVE_DOUBLE);

        std::vector<int> test_indices(100, 0);
        for (unsigned int j = 0; j < 100; ++j) {
            int ref_idx = static_cast<int>(h5_indices[0][j])-1; // object reference index

            // Apply offsets to use correct indices.
            test_indices[j] = static_cast<int>(h5_indices[1][j])-1;
            for (unsigned int k = 0; k < ref_idx; ++k) {
                test_indices[j] += ref_sizes[k];
            }
        }
        test_protocols.push_back(std::move(test_indices));
    }
}
