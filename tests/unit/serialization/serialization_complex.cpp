//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/complex.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

template <typename T>
struct A
{
    A() {}

    A(T t) : t_(t) {}
    T t_;

    A & operator=(T t) { t_ = t; return *this; }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & t_;
    }
};

template <class T>
void test_complex(T real, T imag)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::complex<T> ocomplex(real, imag);
        oarchive << ocomplex;

        hpx::serialization::input_archive iarchive(buffer);
        std::complex<T> icomplex;
        iarchive >> icomplex;
        HPX_TEST_EQ(ocomplex.real(), icomplex.real());
        HPX_TEST_EQ(ocomplex.imag(), icomplex.imag());
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        A<std::complex<T> > ocomplex(std::complex<T>(real, imag));
        oarchive << ocomplex;

        hpx::serialization::input_archive iarchive(buffer);
        A<std::complex<T> > icomplex;
        iarchive >> icomplex;
        HPX_TEST_EQ(ocomplex.t_.real(), icomplex.t_.real());
        HPX_TEST_EQ(ocomplex.t_.imag(), icomplex.t_.imag());
    }
}

int main()
{
    test_complex<float>(100, 100);
    test_complex<double>(100, 100);

    return hpx::util::report_errors();
}
