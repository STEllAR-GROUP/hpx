//  Copyright (c) 2017 Igor Krivenko
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/runtime/serialization/complex.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <complex>
#include <vector>

struct my_struct
{
    using x_type = std::complex<double>;

    my_struct() = delete;
    my_struct(x_type x)
      : x(x)
    {
    }

    bool operator==(my_struct const& ms) const
    {
        return x == ms.x;
    }

    x_type x;

    /// HPX.Serialization
    template <typename Archive>
    inline void serialize(Archive& ar, const unsigned int)
    {
        ar & x;
    }
    template <class Archive>
    inline void friend load_construct_data(
        Archive& ar, my_struct* b, const unsigned int)
    {
        ::new (b) my_struct(0);
    }
};

int main(int argc, char** argv)
{
    std::vector<char> buf;

    // Serialize
    std::vector<my_struct> in;
    in.push_back(my_struct(std::complex<double>(1.0)));
    in.push_back(my_struct(std::complex<double>(2.0)));
    hpx::serialization::output_archive oa(buf);
    oa << in;

    // Deserialize
    std::vector<my_struct> out;
    hpx::serialization::input_archive ia(buf);
    ia >> out;

    HPX_TEST(in == out);

    return hpx::util::report_errors();
}
