#ifndef IDLA__DIFFERENCE_H_
#define IDLA__DIFFERENCE_H_

#include <cassert>

#include <dlib/dnn.h>

#include "difference_impl.h"

namespace idla
{
    /*!
        This object represents a cross-input neighborhood differences layer.
    */
    template <long _nr=5, long _nc=5>
    class cross_neighborhood_differences_ {
    public:
        const static unsigned int sample_expansion_factor = 1;

        static_assert(_nr > 0, "The number of rows in a neighborhood must be > 0");
        static_assert(_nc > 0, "The number of columns in a neighborhood must be > 0");
        static_assert(_nr % 2 != 0, "The number of rows in a neighborhood must be an odd number");
        static_assert(_nc % 2 != 0, "The number of columns in a neighborhood must be an odd number");

        cross_neighborhood_differences_() { }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            DLIB_CASSERT(sub.get_output().num_samples() % 2 == 0);
        }

        /*!
            Performs the forward operation of this layer.
        */
        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
        {
            const dlib::tensor& input_tensor = sub.get_output();
            data_output.set_size(input_tensor.num_samples(), input_tensor.k(),
                                 _nr*input_tensor.nr(), _nc*input_tensor.nc());

#ifdef DLIB_USE_CUDA
            launch_differencing_kernel(input_tensor.device(),
                                       data_output.device_write_only(),
                                       input_tensor.k(),
                                       input_tensor.nr(),
                                       input_tensor.nc(),
                                       _nr,
                                       _nc,
                                       data_output.size()/2);
#else
            COMPILE_TIME_ASSERT("CPU version not implemented yet.")
#endif
        }

        /*!
            Performs the backpropagation step of this layer.
        */
        template <typename SUBNET>
        void backward(
            const dlib::tensor& gradient_input,
            SUBNET& sub,
            dlib::tensor& // params_grad
        )
        {
#ifdef DLIB_USE_CUDA
            const dlib::tensor& input_tensor = sub.get_output();
            launch_differencing_gradient_kernel(gradient_input.device(),
                                                sub.get_gradient_input(),
                                                input_tensor.k(),
                                                input_tensor.nr(),
                                                input_tensor.nc(),
                                                _nr,
                                                _nc,
                                                input_tensor.size());
#else
            COMPILE_TIME_ASSERT("CPU version not implemented yet.")
#endif
        }
    };

    template <long nr, long nc, typename SUBNET>
    using cross_neighborhood_differences = dlib::add_layer<cross_neighborhood_differences_<nr, nc>, SUBNET>;
}



#endif // IDLA__DIFFERENCE_H_
