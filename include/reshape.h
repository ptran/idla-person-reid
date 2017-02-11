#ifndef IDLA__RESHAPE_H_
#define IDLA__RESHAPE_H_

#include <cassert>

#include <dlib/dnn.h>

template <long _n, long _k, long _nr, long _nc>
class reshp_ {
public:
    static_assert(_n > 0, "The number of samples in the reshaped tensor must be > 0");
    static_assert(_k > 0, "The number of channels in the reshaped tensor must be > 0");
    static_assert(_nr > 0, "The number of rows in the reshaped tensor must be > 0");
    static_assert(_nc > 0, "The number of columns in the reshaped tensor must be > 0");

    reshp_() { }

    template <typename SUBNET>
    void setup(const SUBNET& sub)
    {
        // Size constraint
        DLIB_CASSERT(sub.get_output().size() == _n*_k*_nr*_nc, "");
    }

    /*!
        Reshapes the subnet's output to the shape defined in this layer.
    */
    template <typename SUBNET>
    void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
    {
        const dlib::tensor& input_tensor = sub.get_output();
        data_output.set_size(_n, _k, _nr, _nc);
        dlib::memcpy(data_output, input_tensor);
    }

    template <typename SUBNET>
    void backward(
        const dlib::tensor& gradient_input,
        SUBNET& sub,
        dlib::tensor& // params_grad
    )
    {
        const dlib::tensor& input_tensor = sub.get_output();
        dlib::memcpy(sub.get_gradient_input(), input_tensor);
    }

    const dlib::tensor& get_layer_params() const { return params; }
    dlib::tensor& get_layer_params() { return params; }

    friend void serialize(const reshp_& item, std::ostream& out)
    {
        dlib::serialize("reshp", out);
        dlib::serialize(_n, out);
        dlib::serialize(_k, out);
        dlib::serialize(_nr, out);
        dlib::serialize(_nc, out);
    }

    friend void deserialize(reshp_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        long nr;
        long nc;
        if (version == "rshp") {
            dlib::deserialize(nr, in);
            dlib::deserialize(nc, in);
        }
        else {
            throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing reshp_.");
        }

        if (_n != n) throw dlib::serialization_error("Wrong n found while deserializing reshp_");
        if (_k != k) throw dlib::serialization_error("Wrong k found while deserializing reshp_");
        if (_nr != nr) throw dlib::serialization_error("Wrong nr found while deserializing reshp_");
        if (_nc != nc) throw dlib::serialization_error("Wrong nc found while deserializing reshp_");
    }

    friend std::ostream& operator<<(std::ostream& out, const reshp_& item)
    {
        out << "reshp\t ("
            << "n="<<_n
            << ", k="<<_k
            << ", nr="<<_nr
            << ", nc="<<_nc
            << ")";
        return out;
    }

    friend void to_xml(const reshp_& item, std::ostream& out)
    {
        out << "<reshp"
            << " n='"<<_n<<"'"
            << " k='"<<_k<<"'"
            << " nr='"<<_nr<<"'"
            << " nc='"<<_nc<<"'"
            << "/>\n";
    }
private:
    dlib::resizable_tensor params;
};

template <long _n, long _k, long _nr, long _nc, typename SUBNET>
using reshp = dlib::add_layer<reshp_<_n, _k, _nr, _nc>, SUBNET>;

#endif // IDLA__RESHAPE_H_
