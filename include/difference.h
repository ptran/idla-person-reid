#ifndef IDLA__DIFFERENCE_H_
#define IDLA__DIFFERENCE_H_

#include <dlib/dnn.h>

namespace idla {
    namespace impl {
        __global__ void applying_forward_differencing(
            const float* input_tensor,
            float* output_tensor,
            long in_nk,
            long in_nr,
            long in_nc,
            long nbhd_nr,
            long nbhd_nc,
            long n
        );

        __global__ void applying_reverse_differencing(
            const float* input_tensor,
            float* output_tensor,
            long in_nk,
            long in_nr,
            long in_nc,
            long nbhd_nr,
            long nbhd_nc,
            long n
        );

        __global__ void get_gradient(
            const float* input_tensor,
            float* output_tensor,
            long in_nk,
            long in_nr,
            long in_nc,
            long nbhd_nr,
            long nbhd_nc,
            long n
        );
    }

    /*!
        This object represents a cross-input neighborhood difference operation.
    */
    template <long _nr=5, long _nc=5>
    class cross_input_neighborhood_difference_ {
    public:
        const static unsigned int sample_expansion_factor = 2;

        dlib::static_assert(_nr > 0, "The number of rows in a neighborhood must be > 0");
        dlib::static_assert(_nc > 0, "The number of columns in a neighborhood must be > 0");
        dlib::static_assert(_nr % 2 != 0, "The number of rows in a neighborhood must be an odd number");
        dlib::static_assert(_nc % 2 != 0, "The number of columns in a neighborhood must be an odd number");

        cross_input_neighborhood_difference_() { }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            DLIB_CASSERT(input_tensor.num_sample() % 2 == 0);
        }

        /*!
            Performs the forward operation of this layer.
        */
        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
        {
            const dlib::tensor& input_tensor = sub.get_output();
            data_output.set_size(input_tensor.num_samples(), input_tensor.nk(),
                                 _nr*input_tensor.nr(), _nc*input_tensor.nc());
            apply_differencing(input_tensor, data_output);
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
            get_gradient(gradient_input, sub.get_output(), sub.get_gradient_input());
        }
    private:
        void apply_differencing(
            const dlib::tensor& input_tensor,
            dlib::resizable_tensor& data_output
        )
        {
#if DLIB_USE_CUDA
            dlib::launch_kernel(impl::applying_forward_differencing, dlib::max_jobs(data_out.size()),
                                input_tensor.device(),
                                data_output.device_write_only(),
                                input_tensor.nk(), input_tensor.nr(), input_tensor.nc(),
                                _nr, _nc, data_out.size()/2);

            dlib::launch_kernel(impl::applying_backward_differencing, dlib::max_jobs(data_out.size()),
                                input_tensor.device(),
                                data_output.device_write_only() + data_out.size()/2,
                                input_tensor.nk(), input_tensor.nr(), input_tensor.nc(),
                                _nr, _nc, data_out.size()/2);
#else
            COMPILE_TIME_ASSERT("cross_input_neighborhood_difference_.apply_differencing not implemented for CPU.");
#endif
        }

        void get_gradient(
            const dlib::tensor& gradient_input,
            const dlib::tensor& input_tensor,
            dlib::tensor& gradient_output
        )
        {
#if DLIB_USE_CUDA
            dlib::launch_kernel(impl::get_gradient, dlib::max_jobs(gradient_output.size()),
                                gradient_input.device(),
                                gradient_output.device_write_only(),
                                input_tensor.num_samples()/2,
                                input_tensor.nk(), input_tensor.nr(), input_tensor.nc(),
                                _nr, _nc, input_tensor.size());

            COMPILE_TIME_ASSERT("cross_input_neighborhood_difference_.get_gradient not implemented for CPU.");
#endif
        }
#endif
    };
}

#endif // IDLA__DIFFERENCE_H_
