#ifndef IDLA__XNBHD_DIFF_IMPL_CPU_H_
#define IDLA__XNBHD_DIFF_IMPL_CPU_H_

#include <dlib/dnn.h>
#include <dlib/geometry.h>

/*!
    Perform cross neighborhood differencing on the given input tensor.

    @param input_tensor  tensor that cross neighborhood differencing will be 
                         performed on.
    @param output_tensor  tensor that will store the output of applying the 
                          cross neighborhood difference to `input_tensor`.
    @param neighborhood_size  vector with the number of columns (x) and rows (y)
                              in a neighborhood
*/
void perform_cross_neighborhood_differencing(
    const dlib::tensor& input_tensor,
    dlib::resizable_tensor& output_tensor,
    const dlib::vector<long,2>& neighborhood_size
);

/*!
    Backpropagates the gradient of cross neighborhood differencing.

    @param gradient_input  tensor holding the gradient of the successive
                           operation.
    @param gradient_output  tensor that will store the output of backpropagating
                            `gradient_input`
*/
void backpropagate_differencing_gradient(const dlib::tensor& gradient_input, dlib::tensor& gradient_output);

#endif // IDLA__XNBHD_DIFF_IMPL_CPU_H_
