#include <difference.h>

#include <utility>

#include <dlib/dnn.h>

#include "dlib_testing_suite/tester.h"

namespace
{
    using namespace test;

    dlib::logger dlog("test.difference");

    class input_test {
    public:
        typedef dlib::matrix<float> image_type;
        typedef std::pair<image_type*,image_type*> input_type;

        template <typename input_iterator>
        void to_tensor(
            input_iterator ibegin,
            input_iterator iend,
            dlib::resizable_tensor& data
        ) const
        {
            const long nr = ibegin->first->nr();
            const long nc = ibegin->second->nc();
            data.set_size(std::distance(ibegin, iend)*2, 1, nr, nc);

            long offset = nr*nc;
            float* data_ptr = data.host();
            for (auto i = ibegin; i != iend; ++i) {
                for (long r = 0; r < nr; ++r) {
                    for (long c = 0; c < nc; ++c) {
                        float* p = data_ptr++;
                        *p = (*ibegin->first)(r,c);
                        *(p+offset) = (*ibegin->second)(r,c);
                    }
                }
                data_ptr += offset;
            }
        }
    private:
        friend void serialize(const input_test&, std::ostream& out)
        {
            dlib::serialize("input_test", out);
        }

        friend void deserialize(input_test&, std::istream& in)
        {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "input_test") {
                throw dlib::serialization_error("Unexpected version found while deserializing input_test.");
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const input_test&)
        {
            out << "input_test";
            return out;
        }

        friend void to_xml(const input_test&, std::ostream& out)
        {
            out << "<input_test/>";
        }
    };

    class test_difference : public tester {
    public:
        test_difference() : tester("test_difference",
                                   "Runs test on cross neighborhood differences layer")
        { }

        void perform_test()
        {
            dlib::matrix<float> img1(3,3);
            img1 = 1.0;

            dlib::matrix<float> img2(3,3);
            img2 = 0.1, 0.2, 0.3,
                   0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9;

            using net_type = idla::cross_neighborhood_differences<3,3,input_test>;
            net_type net;

            dlib::matrix<float> K1(9,9);
            K1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.7, 0.0,
                 0.0, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4, 0.0,
                 0.0, 0.9, 0.8, 0.9, 0.8, 0.7, 0.8, 0.7, 0.0,
                 0.0, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4, 0.0,
                 0.0, 0.3, 0.2, 0.3, 0.2, 0.1, 0.2, 0.1, 0.0,
                 0.0, 0.6, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4, 0.0,
                 0.0, 0.3, 0.2, 0.3, 0.2, 0.1, 0.2, 0.1, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

            dlib::matrix<float> K2(9,9);
            K2 = 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0,
                 0.0, -0.9, -0.9, -0.8, -0.8, -0.8, -0.7, -0.7, 0.0,
                 0.0, -0.9, -0.9, -0.8, -0.8, -0.8, -0.7, -0.7, 0.0,
                 0.0, -0.6, -0.6, -0.5, -0.5, -0.5, -0.4, -0.4, 0.0,
                 0.0, -0.6, -0.6, -0.5, -0.5, -0.5, -0.4, -0.4, 0.0,
                 0.0, -0.6, -0.6, -0.5, -0.5, -0.5, -0.4, -0.4, 0.0,
                 0.0, -0.3, -0.3, -0.2, -0.2, -0.2, -0.1, -0.1, 0.0,
                 0.0, -0.3, -0.3, -0.2, -0.2, -0.2, -0.1, -0.1, 0.0,
                 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, 0.0;

            std::pair<dlib::matrix<float>*,dlib::matrix<float>*> img_pair = {&img1, &img2};
            dlib::matrix<float> output_mat = dlib::mat(net(img_pair));

            dlib::matrix<float> netK1 = dlib::reshape(dlib::rowm(output_mat, 0), 9, 9);
            DLIB_TEST_MSG(dlib::sum(K1-netK1) <= 1e-4, "");

            dlib::matrix<float> netK2 = dlib::reshape(dlib::rowm(output_mat, 1), 9, 9);
            DLIB_TEST_MSG(dlib::sum(K2-netK2) <= 1e-4, "");
        }
    };

    test_difference a;
}
