#ifndef IDLA__MULTICLASS_LESS_H_
#define IDLA__MULTICLASS_LESS_H_

#include <dlib/dnn.h>

class loss_multiclass_log_lr_  {
public:
#ifdef NEW_DLIB_LOSS    
    typedef unsigned long training_label_type;
    typedef unsigned long output_label_type;
#else
    typedef unsigned long label_type;
#endif

    template <typename SUB_TYPE, typename label_iterator>
    void to_label (
        const dlib::tensor& input_tensor,
        const SUB_TYPE& sub,
        label_iterator iter
    ) const
    {
        const dlib::tensor& output_tensor = sub.get_output();
        DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1 );

        /* REMOVED CASSERTS */
        // DLIB_CASSERT(sub.sample_expansion_factor() == 1);
        // DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

        // Note that output_tensor.k() should match the number of labels.
        for (long i = 0; i < output_tensor.num_samples(); ++i) {
            // The index of the largest output for this sample is the label.
            *iter++ = dlib::index_of_max(dlib::rowm(dlib::mat(output_tensor),i));
        }
    }


    template <typename const_label_iterator, typename SUBNET>
    double compute_loss_value_and_gradient (
        const dlib::tensor& input_tensor,
        const_label_iterator truth, 
        SUBNET& sub
    ) const
    {
        const dlib::tensor& output_tensor = sub.get_output();
        dlib::tensor& grad = sub.get_gradient_input();

        DLIB_CASSERT(input_tensor.num_samples() != 0);
        DLIB_CASSERT(input_tensor.num_samples()%sub.sample_expansion_factor() == 0);
        DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
        DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);

        /* REMOVED CASSERTS */
        // DLIB_CASSERT(sub.sample_expansion_factor() == 1);
        // DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
        // DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());

        dlib::tt::softmax(grad, output_tensor);

        // The loss we output is the average loss over the mini-batch.
        const double scale = 1.0/output_tensor.num_samples();
        double loss = 0;
        float* g = grad.host();
        for (long i = 0; i < output_tensor.num_samples(); ++i) {
            const long y = (long)*truth++;
            // The network must produce a number of outputs that is equal to the number
            // of labels when using this type of loss.
            DLIB_CASSERT(y < output_tensor.k(), "y: " << y << ", output_tensor.k(): " << output_tensor.k());
            for (long k = 0; k < output_tensor.k(); ++k) {
                const unsigned long idx = i*output_tensor.k()+k;
                if (k == y) {
                    loss += scale*-std::log(g[idx]);
                    g[idx] = scale*(g[idx]-1);
                }
                else {
                    g[idx] = scale*g[idx];
                }
            }
        }
        return loss;
    }

    friend void serialize(const loss_multiclass_log_lr_& , std::ostream& out)
    {
        dlib::serialize("loss_multiclass_log_lr_", out);
    }

    friend void deserialize(loss_multiclass_log_lr_& , std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "loss_multiclass_log_lr_")
            throw dlib::serialization_error("Unexpected version found while deserializing dlib::loss_multiclass_log_lr_.");
    }

    friend std::ostream& operator<<(std::ostream& out, const loss_multiclass_log_lr_& )
    {
        out << "loss_multiclass_log_lr";
        return out;
    }

    friend void to_xml(const loss_multiclass_log_lr_& /*item*/, std::ostream& out)
    {
        out << "<loss_multiclass_log_lr/>";
    }
};

template <typename SUBNET>
using loss_multiclass_log_lr = dlib::add_loss_layer<loss_multiclass_log_lr_, SUBNET>;

#endif // IDLA__MULTICLASS_LESS_H_
