#ifndef IDLA_DIFFERENCE_IMPL_H_
#define IDLA_DIFFERENCE_IMPL_H_

/*!
    Calculates cross neighborhood differences.
*/
void launch_differencing_kernel(
    const float* input_tensor,
    float* data_output,
    long in_nk,
    long in_nr,
    long in_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
);

/*!
    Calculates the gradient of cross neighborhood differences.
*/
void launch_differencing_gradient_kernel(
    const float* gradient_input,
    float* gradient_output,
    long in_nk,
    long in_nr,
    long in_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
);

#endif // IDLA_DIFFERENCE_IMPL_H_
