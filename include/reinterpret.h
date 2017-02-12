#ifndef IDLA__REINTERPRET_H_
#define IDLA__REINTERPRET_H_

#include <cassert>

#include <dlib/dnn.h>

/*!
    Reinterprets N-length samples as single samples with N-times the number of 
    channels.
*/
template <long N>
class reinterpret_ {
public:
    static_assert(N > 0, "N must be > 0");

    reinterpret_() { }

    template <typename SUBNET>
    void setup(const SUBNET& sub)
    {
        DLIB_CASSERT(sub.get_output().num_samples() > 0, "");
        DLIB_CASSERT(sub.get_output().num_samples() % N == 0, "");
    }

    template <typename SUBNET>
    void forward(const SUBNET& sub, dlib::resizable_tensor& data_output)
    {
        long n = sub.get_output().num_samples();
        long k = sub.get_output().k();
        long nr = sub.get_output().nr();
        long nc = sub.get_output().nc();
        data_output.set_size(n/N, k*N, nr, nc);

        memcpy(data_output, sub.get_output());
    }

    template <typename SUBNET>
    void backward(
        const dlib::tensor& gradient_input,
        SUBNET& sub,
        dlib::tensor& // params_grad
    )
    {
        dlib::memcpy(sub.get_gradient_input(), gradient_input);
    }

    const dlib::tensor& get_layer_params() const { return params; }
    dlib::tensor& get_layer_params() { return params; }

    friend void serialize(const reinterpret_& item, std::ostream& out)
    {
        dlib::serialize("reshp", out);
        dlib::serialize(N, out);
    }

    friend void deserialize(reinterpret_& item, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        long N_;
        if (version == "reinterpret") {
            dlib::deserialize(N_, in);
        }
        else {
            throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing reinterpret_.");
        }

        if (N != N_) throw dlib::serialization_error("Wrong N found while deserializing reinterpret_");
    }

    friend std::ostream& operator<<(std::ostream& out, const reinterpret_& item)
    {
        out << "reinterpret\t ("
            << "N="<<N
            << ")";
        return out;
    }

    friend void to_xml(const reinterpret_& item, std::ostream& out)
    {
        out << "<reinterpret/"
            << " N='"<<N<<"'"
            << ">\n";
    }
private:
    dlib::resizable_tensor params;
};

template <long N, typename SUBNET>
using reinterpret = dlib::add_layer<reinterpret_<N>, SUBNET>;

#endif // IDLA__REINTERPRET_H_
