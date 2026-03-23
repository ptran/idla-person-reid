#include <difference.h>

#include <utility>

#include <dlib/dnn.h>

#include "input_test.h"
#include "dlib_testing_suite/tester.h"

namespace
{
    using namespace test;

    dlib::logger dlog("test.difference");

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

            using net_type = cross_neighborhood_differences<3,3,input_test>;
            net_type net;

            // =============== //
            //  FORWARD CHECK  //
            // =============== //
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
            DLIB_TEST(dlib::sum(K1-netK1) <= 1e-4);

            dlib::matrix<float> netK2 = dlib::reshape(dlib::rowm(output_mat, 1), 9, 9);
            DLIB_TEST(dlib::sum(K2-netK2) <= 1e-4);

            // ================ //
            //  GRADIENT CHECK  //
            // ================ //
            dlib::resizable_tensor input_tensor;
            input_tensor.set_size(2, 1, 3, 3);
            float* data_ptr = input_tensor.host();
            for (unsigned int i = 0; i < 3; ++i) {
                for (unsigned int j = 0; j < 3; ++j) {
                    *(data_ptr+i*3+j) = img1(i,j);
                    *(data_ptr+9+i*3+j) = img2(i,j);
                }
            }
            net.forward(input_tensor);

            dlib::resizable_tensor gradient_input;
            gradient_input.set_size(2, 1, 9, 9);
            data_ptr = gradient_input.host();
            for (unsigned int i = 1; i < 8; ++i) {
                for (unsigned int j = 1; j < 8; ++j) {
                    *(data_ptr+i*9+j) = 1.0;
                }
            }
            *(data_ptr+91)  = 0.1;  // 0: 4 - 0.1 =  3.9 | 0: 1.6 - 4 = -2.4
            *(data_ptr+92)  = 0.2;  // 1: 6 - 0.2 =  5.8 | 1: 0.0 - 6 = -6.0
            *(data_ptr+97)  = 0.3;  // 2: 4 - 0.3 =  3.7 | 2: 1.6 - 4 = -2.4
            *(data_ptr+100) = 0.5;  // 3: 6 - 0.5 =  5.5 | 3: 5.5 - 6 = -0.5
            *(data_ptr+101) = 0.8;  // 4: 9 - 0.8 =  8.2 | 4: 0.0 - 9 = -9.0
            *(data_ptr+106) = 1.3;  // 5: 6 - 1.3 =  4.7 | 5: 5.5 - 6 = -0.5
            *(data_ptr+127) = 2.1;  // 6: 4 - 2.1 =  1.9 | 6: 0.0 - 4 = -4.0
            *(data_ptr+128) = 3.4;  // 7: 6 - 3.4 =  2.6 | 7: 0.0 - 6 = -6.0
            *(data_ptr+133) = 5.5;  // 8: 4 - 5.5 = -1.5 | 8: 0.0 - 4 = -4.0
            dlib::matrix<float,3,3> grad1, grad2;
            grad1 = 3.9, 5.8,  3.7,
                    5.5, 8.2,  4.7,
                    1.9, 2.6, -1.5;
            grad2 = -2.4, -6.0, -2.4,
                    -0.5, -9.0, -0.5,
                    -4.0, -6.0, -4.0;

            net.back_propagate_error(input_tensor, gradient_input);
            dlib::matrix<float> grad_mat = dlib::mat(net.get_final_data_gradient());

            dlib::matrix<float,3,3> netgrad1 = dlib::reshape(dlib::rowm(grad_mat, 0), 3, 3);
            DLIB_TEST(dlib::sum(grad1-netgrad1) <= 1e-4);

            dlib::matrix<float,3,3> netgrad2 = dlib::reshape(dlib::rowm(grad_mat, 1), 3, 3);
            DLIB_TEST(dlib::sum(grad2-netgrad2) <= 1e-4);
        }
    };

// ---------------------------------------------------------------------------

    test_difference a;
}
