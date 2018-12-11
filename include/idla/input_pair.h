#ifndef IDLA__INPUT_PAIR_H_
#define IDLA__INPUT_PAIR_H_

#include <utility>

#include <dlib/dnn.h>

/*!
  This object represents an input layer that accepts image pairs. The expected
  input types are a pair of pointers to an rgb image.
*/
class input_rgb_image_pair {
public:
    typedef dlib::matrix<dlib::rgb_pixel> image_type;
    typedef std::pair<const image_type, const image_type> input_type;

    /*! @brief  instantiate an input_rgb_image_pair */
    input_rgb_image_pair();

    /*!
      @brief  instantiate an input_rgb_image_pair with pre-computed statistics
      @param avg_red  average red intensity value
      @param avg_green  average green intensity value
      @param avg_blue  average blue intensity value
      @param stddev_red  standard deviation of red intensity values
      @param stddev_green   standard deviation of green intensity values
      @param stddev_blue   standard deviation of blue intensity values
    */
    input_rgb_image_pair(float avg_red, float avg_green, float avg_blue, float stddev_red, float stddev_green, float stddev_blue);

    /*!
      This function converts input image pairs into a data tensor.
    */
    template <typename input_iterator>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const;
private:
    float avg_red = 0;
    float avg_green = 0;
    float avg_blue = 0;
    float stddev_red = 1;
    float stddev_green = 1;
    float stddev_blue = 1;

    friend void serialize(const input_rgb_image_pair& item, std::ostream& out);
    friend void deserialize(input_rgb_image_pair& item, std::istream& in);
    friend std::ostream& operator<<(std::ostream& out, const input_rgb_image_pair& item);
    friend void to_xml(const input_rgb_image_pair& item, std::ostream& out);
};




// =========================================================================== //
//                               IMPLEMENTATION                                //
// =========================================================================== //

template <typename input_iterator>
void input_rgb_image_pair::to_tensor(
    input_iterator ibegin,
    input_iterator iend,
    dlib::resizable_tensor& data
) const
{
    DLIB_CASSERT(std::distance(ibegin, iend) > 0, "Requires at least one example.");

    // Set data tensor size
    const long nr = (ibegin->first).nr();
    const long nc = (ibegin->first).nc();
    data.set_size(std::distance(ibegin, iend)*2, 3, nr, nc);

    long channel_offset = nr*nc;
    long image_offset = 3*channel_offset;

    float* data_ptr = data.host();
    for (auto image_pair = ibegin; image_pair != iend; ++image_pair) {
        auto& img1 = image_pair->first;
        auto& img2 = image_pair->second;
        DLIB_CASSERT(img1.nc() == nc && img2.nc() == nc && img1.nr() == nr && img2.nr() == nr, "Image size mismatch.");
        // Populate data tensor
        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                // Copy the data pointer
                float* p = data_ptr++;
                *p = (static_cast<float>(img1(r, c).red)-avg_red)/stddev_red;
                *(p+image_offset) = (static_cast<float>(img2(r, c).red)-avg_red)/stddev_red;
                p += channel_offset;

                *p = (static_cast<float>(img1(r, c).green)-avg_green)/stddev_green;
                *(p+image_offset) = (static_cast<float>(img2(r, c).green)-avg_green)/stddev_green;
                p += channel_offset;

                *p = (static_cast<float>(img1(r, c).blue)-avg_blue)/stddev_blue;
                *(p+image_offset) = (static_cast<float>(img2(r, c).blue)-avg_blue)/stddev_blue;
            }
        }
        data_ptr += 5*channel_offset;
    }
}

#endif // IDLA__INPUT_PAIR_H_
