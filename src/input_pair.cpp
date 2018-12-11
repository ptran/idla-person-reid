#include "input_pair.h"

input_rgb_image_pair::input_rgb_image_pair()
    : avg_red(0.f),
      avg_green(0.f),
      avg_blue(0.f),
      stddev_red(1.f),
      stddev_green(1.f),
      stddev_blue(1.f)
{ }

input_rgb_image_pair::input_rgb_image_pair(
    float avg_red_,
    float avg_green_,
    float avg_blue_,
    float stddev_red_,
    float stddev_green_,
    float stddev_blue_
) : avg_red(avg_red_),
    avg_green(avg_green_),
    avg_blue(avg_blue_),
    stddev_red(stddev_red_),
    stddev_green(stddev_green_),
    stddev_blue(stddev_blue_)
{ }

void serialize(const input_rgb_image_pair& item, std::ostream& out)
{
    dlib::serialize("input_rgb_image_pair", out);
    dlib::serialize(item.avg_red, out);
    dlib::serialize(item.avg_green, out);
    dlib::serialize(item.avg_blue, out);
    dlib::serialize(item.stddev_red, out);
    dlib::serialize(item.stddev_green, out);
    dlib::serialize(item.stddev_blue, out);
}

void deserialize(input_rgb_image_pair& item, std::istream& in)
{
    std::string version;
    dlib::deserialize(version, in);
    if (version != "input_rgb_image_pair") {
        throw dlib::serialization_error("Unexpected version found while deserializing input_rgb_image_pair.");
    }
    dlib::deserialize(item.avg_red, in);
    dlib::deserialize(item.avg_green, in);
    dlib::deserialize(item.avg_blue, in);
    dlib::deserialize(item.stddev_red, in);
    dlib::deserialize(item.stddev_green, in);
    dlib::deserialize(item.stddev_blue, in);
}

std::ostream& operator<<(std::ostream& out, const input_rgb_image_pair& item)
{
    out << "input_rgb_image_pair";
    return out;
}

void to_xml(const input_rgb_image_pair& item, std::ostream& out)
{
    out << "<input_rgb_image_pair/>";
}
