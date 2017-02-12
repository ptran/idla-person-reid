#ifndef IDLA_TEST__INPUT_TEST
#define IDLA_TEST__INPUT_TEST

// An input layer for testing
class input_test {
public:
    typedef dlib::matrix<float> image_type;
    typedef std::pair<image_type*,image_type*> input_type;

    template <typename input_iterator>
    void to_tensor(
        input_iterator ibegin,
        input_iterator iend,
        dlib::resizable_tensor& data
    ) const
    {
        const long nr = ibegin->first->nr();
        const long nc = ibegin->second->nc();
        data.set_size(std::distance(ibegin, iend)*2, 1, nr, nc);

        long offset = nr*nc;
        float* data_ptr = data.host();
        for (auto i = ibegin; i != iend; ++i) {
            for (long r = 0; r < nr; ++r) {
                for (long c = 0; c < nc; ++c) {
                    float* p = data_ptr++;
                    *p = (*ibegin->first)(r,c);
                    *(p+offset) = (*ibegin->second)(r,c);
                }
            }
            data_ptr += offset;
        }
    }
private:
    friend void serialize(const input_test&, std::ostream& out)
    {
        dlib::serialize("input_test", out);
    }

    friend void deserialize(input_test&, std::istream& in)
    {
        std::string version;
        dlib::deserialize(version, in);
        if (version != "input_test") {
            throw dlib::serialization_error("Unexpected version found while deserializing input_test.");
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const input_test&)
    {
        out << "input_test";
        return out;
    }

    friend void to_xml(const input_test&, std::ostream& out)
    {
        out << "<input_test/>";
    }
};

#endif // IDLA_TEST__INPUT_TEST
