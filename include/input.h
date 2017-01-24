#include <pair>

#include <dlib/dnn.h>


template <typename image_type>
class input_image_pair {
    typedef dlib::image_traits<image_type>::pixel_type pixel_type;
public:
    const static unsigned int sample_expansion_factor = 2;
    typedef std::pair<image_type*,image_type*> input_type;

    template <typename input_iterator>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const
    {
        DLIB_CASSERT(std::distance(ibegin, iend) > 0,"");
        // Set data tensor size
        {
            dlib::const_image_view<image_type> img(*(ibegin->first));
            const long nk = dlib::pixel_traits<pixel_type>.num;
            const long nr = img.nr();
            const long nc = img.nc();
            data.set_size(std::distance(ibegin, iend)*2, nk, nr, nc);
        }

        for (auto i = ibegin; i != iend; ++i) {
            dlib::const_image_view<image_type> img1(*(i->first));
            dlib::const_image_view<image_type> img2(*(i->second));
            DLIB_CASSERT(img1.nc() == nc && img2.nc() == nc &&
                         img1.nr() == nr && img2.nr() == nr,
                         "Image size mismatch.");
        }

        long offset = nr*nc;
        float* data_ptr = data.host();
        for (auto i = ibegin; i != iend; ++i) {
            for (long r = 0; r < nr; ++r) {
                for (long c = 0; c < nc; ++c) {
                    // Copy the data pointer while also iterating to the next
                    // element.
                    float* p = data_ptr++;
                    *p = static_cast<float>(*(i->first)(r,c))/256.0;
                    *(p+offset) = static_cast<float>(*(i->second)(r,c))/256.0;
                }
            }
            data_ptr += offset;
        }
    }
};

void serialize(const input_image_pair& item, std::ostream& out)
{
    dlib::serialize("input_image_pair", out);
}

void deserialize(input_image_pair& item, std::istream& in)
{
    std::string version;
    dlib::deserialize(version, in);
    if (version != "input_image_pair") {
        throw dlib::serialization_error("Unexpected version found while deserializing input_image_pair.");
    }
}

std::ostream& operator<<(std::ostream& out, const input_image_pair& item)
{
    out << "input_image_pair";
    return out;
}

void to_xml(const input_image_pair& item, std::ostream& out)
{
    out << "<input_image_pair/>";
}
