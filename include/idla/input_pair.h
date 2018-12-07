#ifndef IDLA__INPUT_PAIR_H_
#define IDLA__INPUT_PAIR_H_

#include <utility>

#include <dlib/statistics.h>
#include <dlib/dnn.h>

/*!
    This object represents an input layer that accepts image pairs. The expected
    input types are a pair of pointers to an rgb image.
*/
class input_rgb_image_pair {
public:
    typedef dlib::matrix<dlib::rgb_pixel> image_type;
    typedef std::pair<const image_type*,const image_type*> input_type;

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
    const long nr = (*ibegin->first).nr();
    const long nc = (*ibegin->first).nc();
    data.set_size(std::distance(ibegin, iend)*2, 3, nr, nc);

    long channel_offset = nr*nc;
    long image_offset = 3*channel_offset;

    float* data_ptr = data.host();
    for (auto i = ibegin; i != iend; ++i) {
        DLIB_CASSERT((*i->first).nc() == nc && (*i->second).nc() == nc &&
                     (*i->first).nr() == nr && (*i->second).nr() == nr,
                     "Image size mismatch.");

        // Find image statistics for normalization
        dlib::running_stats<float> stats1, stats2;
        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                stats1.add((*i->first)(r,c).red);
                stats1.add((*i->first)(r,c).green);
                stats1.add((*i->first)(r,c).blue);
                stats2.add((*i->second)(r,c).red);
                stats2.add((*i->second)(r,c).green);
                stats2.add((*i->second)(r,c).blue);
            }
        }

        // Populate data tensor
        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                // Copy the data pointer
                float* p = data_ptr++;
                dlib::rgb_pixel tmp1 = (*i->first)(r,c);
                dlib::rgb_pixel tmp2 = (*i->second)(r,c);

                *p = (static_cast<float>(tmp1.red)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+image_offset) = (static_cast<float>(tmp2.red)-stats2.mean())/(stats2.stddev()+1e-7);
                p += channel_offset;

                *p = (static_cast<float>(tmp1.green)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+image_offset) = (static_cast<float>(tmp2.green)-stats2.mean())/(stats2.stddev()+1e-7);
                p += channel_offset;

                *p = (static_cast<float>(tmp1.blue)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+image_offset) = (static_cast<float>(tmp2.blue)-stats2.mean())/(stats2.stddev()+1e-7);
            }
        }
        data_ptr += 5*channel_offset;
    }
}

#endif // IDLA__INPUT_PAIR_H_
