#include <utility>

#include <dlib/statistics.h>
#include <dlib/dnn.h>

/*!
    This object represents an input layer that accepts image pairs. The expected
    input types are a pair of pointers to an rgb image.
*/
class input_rgb_image_pair {
public:
    const static unsigned int sample_expansion_factor = 2;
    typedef dlib::matrix<dlib::rgb_pixel> image_type;
    typedef std::pair<image_type*,image_type*> input_type;

    template <typename input_iterator>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const;
};

void serialize(const input_rgb_image_pair& item, std::ostream& out);

void deserialize(input_rgb_image_pair& item, std::istream& in);

std::ostream& operator<<(std::ostream& out, const input_rgb_image_pair& item);

void to_xml(const input_rgb_image_pair& item, std::ostream& out);




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
    const long nc = (*ibegin->first).nr();
    data.set_size(std::distance(ibegin, iend)*2, 3, nr, nc);
    
    long offset = nr*nc*3;
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
                // Copy the data pointer while also iterating to the next
                // element.
                float* p = data_ptr++;
                *p = (static_cast<float>((*i->first)(r,c).red)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+offset) = (static_cast<float>((*i->second)(r,c).red)-stats2.mean())/(stats2.stddev()+1e-7);
                ++p;
                *p = (static_cast<float>((*i->first)(r,c).green)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+offset) = (static_cast<float>((*i->second)(r,c).green)-stats2.mean())/(stats2.stddev()+1e-7);
                ++p;
                *p = (static_cast<float>((*i->first)(r,c).blue)-stats1.mean())/(stats1.stddev()+1e-7);
                *(p+offset) = (static_cast<float>((*i->second)(r,c).blue)-stats2.mean())/(stats2.stddev()+1e-7);
            }
        }
        data_ptr += offset;
    }
}
