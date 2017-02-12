#include <reinterpret.h>

#include <dlib/dnn.h>

#include "input_test.h"
#include "dlib_testing_suite/tester.h"

namespace
{
    using namespace test;

    dlib::logger dlog("test.reinterpret");

    class test_reinterpret : public tester {
    public:
        test_reinterpret() : tester("test_reinterpret",
                                "Runs test on reinterpret layer")
        { }

        void perform_test()
        {
            dlib::matrix<float> img1(3,3);
            img1 = 1337.0;

            dlib::matrix<float> img2(3,3);
            img2 = 8008.0;

            using net_type = reinterpret<2,input_test>;
            net_type net;

            std::pair<dlib::matrix<float>*,dlib::matrix<float>*> img_pair = {&img1, &img2};
            dlib::resizable_tensor output = net(img_pair);
            DLIB_TEST(output.num_samples() == 1 && output.k() == 2);
        }
    };

    test_reinterpret a;
}
