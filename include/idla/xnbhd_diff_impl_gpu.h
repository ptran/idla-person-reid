#ifndef IDLA__XNBHD_DIFF_IMPL_GPU_H_
#define IDLA__XNBHD_DIFF_IMPL_GPU_H_

/*!
    Kernel that performs cross neighborhood differencing.

    @param input_tensor  pointer to an input tensor that cross neighborhood 
                         differencing will be performed on.
    @param data_output  pointer to a tensor that will store the output of 
                        applying the cross neighborhood difference to 
                        `input_tensor`.
    @param in_nk  number of channels in input tensor.
    @param in_nr  number of rows in input tensor.
    @param in_nc  number of columns in input tensor.
    @param nbhd_nr  number of rows in a neighborhood.
    @param nbhd_nc  number of columns in a neighborhood.
    @param n  number of input tensor elements.
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
    Kernel that backpropagates the gradient of cross neighborhood differencing.

    @param gradient_input  pointer to an input tensor holding the gradient of 
                           the successive operation.
    @param gradient_output  pointer to a tensor that will store the output of 
                            backpropagating `gradient_input`
    @param in_nk  number of channels in input tensor.
    @param in_nr  number of rows in input tensor.
    @param in_nc  number of columns in input tensor.
    @param nbhd_nr  number of rows in a neighborhood.
    @param nbhd_nc  number of columns in a neighborhood.
    @param n  number of input tensor elements.
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

#endif // IDLA__XNBHD_DIFF_IMPL_GPU_H_
