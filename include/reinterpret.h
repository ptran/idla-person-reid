#ifndef IDLA__REINTERPRET_H_
#define IDLA__REINTERPRET_H_

#include <cassert>

#include <dlib/dnn.h>

#ifdef DLIB_USE_CUDA
#include <cuda_runtime.h>
#endif

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

#ifdef DLIB_USE_CUDA
        cudaMemcpy(data_output.device(), sub.get_output().device(), data_output.size()*sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(data_output.host(), sub.get_output().host(), data_output.size()*sizeof(float));
#endif
    }

    template <typename SUBNET>
    void backward(
        const dlib::tensor& gradient_input,
        SUBNET& sub,
        dlib::tensor& // params_grad
    )
    {
#ifdef DLIB_USE_CUDA
        cudaMemcpy(sub.get_gradient_input().device(), gradient_input.device(), gradient_input.size()*sizeof(float), cudaMemcpyDeviceToDevice);
#else
        memcpy(sub.get_gradient_input().host(), gradient_input.host(), gradient_input.size()*sizeof(float));
#endif
    }

    const dlib::tensor& get_layer_params() const { return params; }
    dlib::tensor& get_layer_params() { return params; }

    friend void serialize(const reinterpret_& item, std::ostream& out)
    {
        dlib::serialize("reinterpret", out);
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
