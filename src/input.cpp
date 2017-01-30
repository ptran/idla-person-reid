#include "input.h"

void serialize(const input_rgb_image_pair& item, std::ostream& out)
{
    dlib::serialize("input_rgb_image_pair", out);
}

void deserialize(input_rgb_image_pair& item, std::istream& in)
{
    std::string version;
    dlib::deserialize(version, in);
    if (version != "input_rgb_image_pair") {
        throw dlib::serialization_error("Unexpected version found while deserializing input_rgb_image_pair.");
    }
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
