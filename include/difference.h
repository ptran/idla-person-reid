#ifndef IDLA__DIFFERENCE_H_
#define IDLA__DIFFERENCE_H_

#include <cassert>

#include <dlib/dnn.h>

#include "difference_impl.h"

/*!
    This object represents a cross-input neighborhood differences layer.
*/
template <long _nr=5, long _nc=5>
class cross_neighborhood_differences_ {
public:
    const static unsigned int sample_expansion_factor = 1;

    static_assert(_nr > 0, "The number of rows in the neighborhood region must be > 0");
    static_assert(_nc > 0, "The number of columns in the neighborhood region must be > 0");
    static_assert(_nr % 2 != 0, "The number of rows in the neighborhood region must be an odd number");
    static_assert(_nc % 2 != 0, "The number of columns in the neighborhood region must be an odd number");

    cross_neighborhood_differences_() { }

    template <typename SUBNET>
    void setup(const SUBNET& sub)
    {
        DLIB_CASSERT(sub.get_output().num_samples() % 2 == 0, "");
    }

    /*!
        Performs the cross-input neighborhood differencing operation to each 
        sample pair.
    */
    template <typename SUBNET>
    void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
    {
        // Cross-input neighborhood differencing creates an output that has _nr
        // times more rows and _nc times more columns. This is due to the
        // neighborhood output produced at every pixel.
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
                                   data_output.size());
#else
        COMPILE_TIME_ASSERT("CPU version not implemented yet.");
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
                                            sub.get_gradient_input().device(),
                                            input_tensor.k(),
                                            input_tensor.nr(),
                                            input_tensor.nc(),
                                            _nr,
                                            _nc,
                                            input_tensor.size());
#else
        COMPILE_TIME_ASSERT("CPU version not implemented yet.");
#endif
    }

    const dlib::tensor& get_layer_params() const { return params; }
    dlib::tensor& get_layer_params() { return params; }

    friend void serialize(const cross_neighborhood_differences_& item, std::ostream& out)
    {
        dlib::serialize("cross_neighborhood_differences", out);
        dlib::serialize(_nr, out);
        dlib::serialize(_nc, out);
    }

    friend void deserialize(cross_neighborhood_differences_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        long nr;
        long nc;
        if (version == "cross_neighborhood_differences") {
            dlib::deserialize(nr, in);
            dlib::deserialize(nc, in);
        }
        else {
            throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing cross_neighborhood_differences_.");
        }

        if (_nr != nr) throw dlib::serialization_error("Wrong nr found while deserializing cross_neighborhood_differences_");
        if (_nc != nc) throw dlib::serialization_error("Wrong nc found while deserializing cross_neighborhood_differences_");
    }

    friend std::ostream& operator<<(std::ostream& out, const cross_neighborhood_differences_& item)
    {
        out << "cross_neighborhood_differences\t ("
            << "nr="<<_nr
            << ", nc="<<_nc
            << ")";
        return out;
    }

    friend void to_xml(const cross_neighborhood_differences_& item, std::ostream& out)
    {
        out << "<cross_neighborhood_differences"
            << " nr='"<<_nr<<"'"
            << " nc='"<<_nc<<"'"
            << "/>\n";
    }
private:
    dlib::resizable_tensor params;
};

template <long nr, long nc, typename SUBNET>
using cross_neighborhood_differences = dlib::add_layer<cross_neighborhood_differences_<nr, nc>, SUBNET>;

#endif // IDLA__DIFFERENCE_H_
