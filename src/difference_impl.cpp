#include "difference_impl_cpu.h"

namespace
{
    float* get_element_pointer(dlib::tensor& T, long n, long k, long r, long c)
    {
        return T.host() + ((n*T.k()+k)*T.nr()+r)*T.nc() + c;
    }

    const float* get_element_pointer(const dlib::tensor& T, long n, long k, long r, long c)
    {
        return T.host() + ((n*T.k()+k)*T.nr()+r)*T.nc() + c;
    }
}

void perform_cross_neighborhood_differencing(
    const dlib::tensor& input_tensor,
    dlib::resizable_tensor& output_tensor,
    const dlib::vector<long,2>& neighborhood_size
)
{
    long nbhd_nc = neighborhood_size.x();
    long nbhd_nr = neighborhood_size.y();

    for (long n = 0; n < input_tensor.num_samples(); ++n) {
        long flag = (n % 2 == 0) ? 1 : -1;
        for (long k = 0; k < input_tensor.k(); ++k) {
            for (long r = 0; r < input_tensor.nr(); ++r) {
                for (long c = 0; c < input_tensor.nc(); ++c) {
                    float comparison_pixel = *get_element_pointer(input_tensor, n, k, r, c);
                    for (long nbhd_r = 0; nbhd_r < nbhd_nr; ++nbhd_r) {
                        long img_r = r - nbhd_nr/2 + nbhd_r;
                        for (long nbhd_c = 0; nbhd_c < nbhd_nc; ++nbhd_c) {
                            long img_c = c - nbhd_nc/2 + nbhd_c;
                            float* out_ptr = get_element_pointer(output_tensor, n, k, r*nbhd_nr + nbhd_r, c*nbhd_nc + nbhd_c);
                            
                            if (img_r < 0 || img_r >= input_tensor.nr() || img_c < 0 || img_c >= input_tensor.nc()) {
                                *out_ptr = 0.0f;
                            } else {
                                *out_ptr = comparison_pixel - *get_element_pointer(input_tensor, n+flag, k, img_r, img_c);
                            }
                        }
                    }
                }
            }
        }
    }
}

void backpropagate_differencing_gradient(const dlib::tensor& gradient_input, dlib::tensor& gradient_output)
{
    long in_nr = gradient_output.nr();
    long in_nc = gradient_output.nc();
    long nbhd_nr = gradient_input.nr() / in_nr;
    long nbhd_nc = gradient_input.nc() / in_nc;

    // We MUST accumulate (+=) gradients to respect dlib's internal architecture.
    for (long n = 0; n < gradient_output.num_samples(); ++n) {
        long flag = (n % 2 == 0) ? 1 : -1;
        for (long k = 0; k < gradient_output.k(); ++k) {
            for (long r = 0; r < in_nr; ++r) {
                for (long c = 0; c < in_nc; ++c) {
                    
                    float* grad_out_comparison = get_element_pointer(gradient_output, n, k, r, c);

                    for (long nbhd_r = 0; nbhd_r < nbhd_nr; ++nbhd_r) {
                        long img_r = r - nbhd_nr/2 + nbhd_r;
                        for (long nbhd_c = 0; nbhd_c < nbhd_nc; ++nbhd_c) {
                            long img_c = c - nbhd_nc/2 + nbhd_c;
                            
                            float grad_in = *get_element_pointer(gradient_input, n, k, r*nbhd_nr + nbhd_r, c*nbhd_nc + nbhd_c);

                            if (img_r >= 0 && img_r < in_nr && img_c >= 0 && img_c < in_nc) {
                                // Accumulate contribution to comparison pixel (Positive)
                                *grad_out_comparison += grad_in;
                                
                                // Accumulate contribution to neighborhood pixel (Negative)
                                float* grad_out_nbhd = get_element_pointer(gradient_output, n+flag, k, img_r, img_c);
                                *grad_out_nbhd -= grad_in;
                            }
                        }
                    }
                }
            }
        }
    }
}
