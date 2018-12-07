#include "xnbhd_diff_impl_cpu.h"

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

    // Iterate through each dimension of the tensor
    for (long n = 0; n < input_tensor.num_samples(); ++n) {
        // Flag that determines the sample offset for the "neighborhood image"
        long flag = (n % 2 == 0) ? 1 : -1;
        for (long k = 0; k < input_tensor.k(); ++k) {
            for (long r = 0; r < input_tensor.nr(); ++r) {
                for (long c = 0; c < input_tensor.nc(); ++c) {
                    // Get central comparison pixel
                    float comparison_pixel = *get_element_pointer(input_tensor, n, k, r, c);
                    // Iterate through the neighborhood
                    for (long nbhd_r = 0; nbhd_r < nbhd_nr; ++nbhd_r) {
                        // Get output pointer without the column offset
                        float* output_ptr_no_c = get_element_pointer(output_tensor, n, k, r*nbhd_nr + nbhd_r, 0);
                        long img_r = r - nbhd_nr/2 + nbhd_r;  // image row position
                        for (long nbhd_c = 0; nbhd_c < nbhd_nc; ++nbhd_c) {
                            float* output_ptr = output_ptr_no_c + c*nbhd_nc + nbhd_c;
                            // If the current row is out of bounds...
                            if (img_r < 0 || img_r >= input_tensor.nr()) {
                                *output_ptr = 0.0;
                                continue;
                            }
                            // If the current column is out of bounds...
                            long img_c = c - nbhd_nc/2 + nbhd_c; // image column position
                            if (img_c < 0 || img_c >= input_tensor.nc()) {
                                *output_ptr = 0.0;
                                continue;
                            }
                            // Perform differencing
                            *output_ptr = comparison_pixel - *get_element_pointer(input_tensor, n+flag, k, img_r, img_c);
                        }
                    }
                }
            }
        }
    }
}

void backpropagate_differencing_gradient(const dlib::tensor& gradient_input, dlib::tensor& gradient_output)
{
    long nbhd_nr = gradient_input.nr()/gradient_output.nr();
    long nbhd_nc = gradient_input.nc()/gradient_output.nc();

    // Iterate through each dimension of the tensor
    for (long n = 0; n < gradient_output.num_samples(); ++n) {
        // Flag that determines the sample offset for the "neighborhood image"
        long flag = (n % 2 == 0) ? 1 : -1;
        for (long k = 0; k < gradient_output.k(); ++k) {
            for (long r = 0; r < gradient_output.nr(); ++r) {
                for (long c = 0; c < gradient_output.nc(); ++c) {
                    float* output_ptr = get_element_pointer(gradient_output, n, k, r, c);
                    *output_ptr = 0.0;

                    // Iterate through the neighborhood and accumulate gradients
                    // for when the current pixel was the central comparison
                    // pixel and a "neighborhood image" pixel
                    for (long nbhd_r = 0; nbhd_r < nbhd_nr; ++nbhd_r) {
                        const float* input_ptr1 = nullptr;
                        long img_r1 = r + (nbhd_r-nbhd_nr/2);
                        if (img_r1 >= 0 && img_r1 < gradient_output.nr()) {
                            input_ptr1 = get_element_pointer(gradient_input, n, k, r*nbhd_nr + nbhd_r, 0);
                        }

                        const float* input_ptr2 = nullptr;
                        long scan_r = r + nbhd_nr/2 - nbhd_r; // neighborhood image row
                        long img_r2 = scan_r + (nbhd_r-nbhd_nr/2);
                        if (img_r2 >= 0 && img_r2 < gradient_output.nr()) {
                            input_ptr2 = get_element_pointer(gradient_input, n+flag, k, scan_r*nbhd_nr + nbhd_r, 0);
                        }

                        for (long nbhd_c = 0; nbhd_c < nbhd_nc; ++nbhd_c) {
                            if (input_ptr1 != nullptr) {
                                long img_c1 = c + (nbhd_c-nbhd_nc/2);
                                if (img_c1 >= 0 && img_c1 < gradient_output.nc()) {
                                    *output_ptr += *(input_ptr1 + c*nbhd_nc + nbhd_c);
                                }
                            }
                            if (input_ptr2 != nullptr) {
                                long scan_c = c + nbhd_nc/2 - nbhd_c; // neighborhood image column
                                long img_c2 = scan_c + (nbhd_c-nbhd_nc/2);
                                if (img_c2 >= 0 && img_c2 < gradient_output.nc()) {
                                    *output_ptr -= *(input_ptr2 + scan_c*nbhd_nc + nbhd_c);
                                }
                            }
                        } // nbhd_c
                    } // nbhd_r
                } // c
            } // r
        } // k
    } // n
}
