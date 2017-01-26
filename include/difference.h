#ifndef IDLA__DIFFERENCE_H_
#define IDLA__DIFFERENCE_H_

#include <cassert>

#include <dlib/dnn.h>

namespace idla {

    namespace impl {
        void apply_differencing(
            const dlib::tensor& input_tensor,
            dlib::resizable_tensor& data_output,
            long nbhd_nr,
            long nbhd_nc
        );

        void get_differencing_gradient(
            long nbhd_nr,
            long nbhd_nc
        );
    }

    /*!
        This object represents a cross-input neighborhood differences layer.
    */
    template <long _nr=5, long _nc=5>
    class cross_input_neighborhood_differences_ {
    public:
        const static unsigned int sample_expansion_factor = 2;

        static_assert(_nr > 0, "The number of rows in a neighborhood must be > 0");
        static_assert(_nc > 0, "The number of columns in a neighborhood must be > 0");
        static_assert(_nr % 2 != 0, "The number of rows in a neighborhood must be an odd number");
        static_assert(_nc % 2 != 0, "The number of columns in a neighborhood must be an odd number");

        cross_input_neighborhood_differences_() { }

        template <typename SUBNET>
        void setup(const SUBNET& sub)
        {
            DLIB_CASSERT(sub.get_output.num_sample() % 2 == 0);
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
            impl::apply_differencing(input_tensor, data_output, _nr, _nc);
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
            impl::get_differencing_gradient(gradient_input, sub.get_output(), sub.get_gradient_input());
        }
    };

    template <long nr, long nc, typename SUBNET>
    using cross_input_neighborhood_differences =
        dlib::add_layer<cross_input_neighborhood_differences_<nr, nc>, SUBNET>;
}



#endif // IDLA__DIFFERENCE_H_
