#include <multiclass_loss.h>
#include <dlib/dnn.h>
#include "dlib_testing_suite/tester.h"

namespace
{
    using namespace test;

    class test_multiclass_loss : public tester {
    public:
        test_multiclass_loss() : tester("test_multiclass_loss",
                                     "Runs tests on the custom multiclass loss layer.")
        { }

        void perform_test()
        {
            using net_type = loss_multiclass_log_lr<dlib::fc<2, dlib::input<dlib::matrix<float>>>>;
            net_type net;

            // Create a simple input
            dlib::matrix<float> input(1, 1);
            input = 0.5f;

            std::vector<dlib::matrix<float>> inputs = {input};
            std::vector<unsigned long> labels = {1};

            // Before training, get the loss
            double loss1 = net.compute_loss(inputs.begin(), inputs.end(), labels.begin());
            DLIB_TEST(loss1 > 0);

            // Train for several steps to ensure a decrease
            dlib::dnn_trainer<net_type> trainer(net);
            trainer.set_learning_rate(0.1);
            trainer.set_min_learning_rate(0.001);
            for (int i = 0; i < 10; ++i)
                trainer.train_one_step(inputs.begin(), inputs.end(), labels.begin());

            // After training, the loss should have decreased
            double loss2 = net.compute_loss(inputs.begin(), inputs.end(), labels.begin());
            DLIB_TEST_MSG(loss2 < loss1, "Loss did not decrease: " << loss1 << " -> " << loss2);
        }
    };

    test_multiclass_loss a;
}
