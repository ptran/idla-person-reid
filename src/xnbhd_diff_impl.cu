#include "xnbhd_diff_impl_gpu.h"
#include <sstream>

__global__ void apply_differencing_impl(
    const float* input_tensor,
    float* output_tensor,
    long in_nk,
    long in_nr,
    long in_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
)
{
    // Current GPU index
    long i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i > n) { return; }

    // Find neighborhood indices (center comparison pixel location)
    long nbhd_c = i/nbhd_nc % in_nc;               // also center column
    long nbhd_r = i/nbhd_nc/in_nc/nbhd_nr % in_nr; // also center row
    long k = i/nbhd_nc/in_nc/nbhd_nr/in_nr % in_nk;
    long sample = i/nbhd_nc/in_nc/nbhd_nr/in_nr/in_nk;

    // Find in-neighborhood row and column indices
    long in_nbhd_c = i % nbhd_nc;
    long in_nbhd_r = i/nbhd_nc/in_nc % nbhd_nr;

    // Find the "neighborhood image" row and column indices
    long in_c = nbhd_c - nbhd_nc/2 + in_nbhd_c;
    long in_r = nbhd_r - nbhd_nr/2 + in_nbhd_r;

    // `flag` flips which image in the image pair is the "neighborhood image".
    long flag = (sample % 2 == 0) ? 1 : -1;
    // If the neighborhood indexing exceeds the size of the "neighborhood image", set the difference output to 0
    // (no activation).
    if (in_c < 0 || in_r < 0 || in_nc <= in_c ||  in_nr <= in_r) {
        output_tensor[i] = 0.0;
    }
    else {
        long idx1 = ((sample*in_nk + k)*in_nr + nbhd_r)*in_nc + nbhd_c;
        long idx2 = (((sample+flag)*in_nk + k)*in_nr + in_r)*in_nc + in_c;
        output_tensor[i] = input_tensor[idx1]-input_tensor[idx2];
    }
}

__global__ void get_differencing_gradient_impl(
    const float* gradient_input,
    float* gradient_output,
    long out_nk,
    long out_nr,
    long out_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
)
{
    // Current GPU index
    long i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i > n) { return; }

    // Find the output indices
    long out_c = i % out_nc;
    long out_r = i/out_nc % out_nr;
    long k = i/out_nc/out_nr % out_nk;
    long sample = i/out_nc/out_nr/out_nk;

    // Backpropagate gradients for when the current pixel was the center
    // comparison pixel.
    gradient_output[i] = 0;
    for (long r = out_r*nbhd_nr; r < (out_r+1)*nbhd_nr; ++r) {
        long offset = ((sample*out_nk + k)*out_nr*nbhd_nr + r)*out_nc*nbhd_nc;
        for (long c = out_c*nbhd_nc; c < (out_c+1)*nbhd_nc; ++c) {
            gradient_output[i] += gradient_input[offset + c];
        }
    }

    long flag = (sample % 2 == 0) ? 1 : -1;
    long r_off = nbhd_nr/2;
    long c_off = nbhd_nc/2;

    // Backpropagate gradients for when the current pixel was part of the
    // "neighborhood image"
    long out_nbhd_r = 0;  // in-neighborhood row index
    for (long r = out_r+r_off; r >= out_r-r_off; --r) {
        if (r < 0 || r >= out_nr) {
            ++out_nbhd_r;
            continue;
        }
        long out_nbhd_c = 0;  // in-neighborhood column index
        long offset = (((sample+flag)*out_nk + k)*out_nr*nbhd_nr + r*nbhd_nr + out_nbhd_r)*out_nc*nbhd_nc;
        ++out_nbhd_r;

        for (long c = out_c+c_off; c >= out_c-c_off; --c) {
            if (c < 0 || c >= out_nc) {
                ++out_nbhd_c;
                continue;
            }
            gradient_output[i] -= gradient_input[offset + c*nbhd_nc + out_nbhd_c];
            ++out_nbhd_c;
        }
    }
}

void launch_differencing_kernel(
    const float* input_tensor,
    float* data_output,
    long in_nk,
    long in_nr,
    long in_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
)
{
    int nblocks;
    int nthreads;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&nblocks, &nthreads, apply_differencing_impl);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error when calling launch_differencing_kernel"
            << "\ncode: " << err << ", error: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
    apply_differencing_impl<<<nblocks, nthreads>>>(input_tensor, data_output, in_nk, in_nr, in_nc, nbhd_nr, nbhd_nc, n);
}

void launch_differencing_gradient_kernel(
    const float* gradient_input,
    float* gradient_output,
    long in_nk,
    long in_nr,
    long in_nc,
    long nbhd_nr,
    long nbhd_nc,
    long n
)
{
    int nblocks;
    int nthreads;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&nblocks, &nthreads, get_differencing_gradient_impl);
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error when calling launch_differencing_gradient_kernel"
            << "\ncode: " << err << ", error: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
    get_differencing_gradient_impl<<<nblocks, nthreads>>>(gradient_input, gradient_output,
                                                          in_nk, in_nr, in_nc,
                                                          nbhd_nr, nbhd_nc,
                                                          n);
}
