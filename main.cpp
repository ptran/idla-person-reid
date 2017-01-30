#include <iostream>

#include <dlib/matrix.h>
#include <dlib/pixel.h>

#include "input.h"
#include "difference.h"

int main(int argc, char* argv[])
{
    std::cout << "Just glassin'" << std::endl;

    using net_type = idla::cross_neighborhood_differences<3,3,input_rgb_image_pair>;
    net_type net;

    dlib::matrix<dlib::rgb_pixel> img1(3,3);
    dlib::matrix<dlib::rgb_pixel> img2(3,3);
    img1 = dlib::rgb_pixel(10,3,10);
    img2 = dlib::rgb_pixel(5,8,5);

    input_rgb_image_pair::input_type in1 = {&img1, &img2};
    dlib::resizable_tensor output = net(in1);
    dlib::matrix<float> outmat = dlib::mat(output);

    std::cout << output.num_samples() << ","
              << output.k()           << ","
              << output.nr()          << ","
              << output.nc()          << std::endl;
    return 0;
}
