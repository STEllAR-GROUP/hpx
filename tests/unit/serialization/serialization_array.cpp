//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/multi_array.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <array>
#include <cstddef>
#include <numeric>
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

template <typename T>
void test(T minval, T maxval)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::vector<T> os;
        for(T c = minval; c < maxval; ++c)
        {
            os.push_back(c);
        }
        oarchive << hpx::serialization::make_array(&os[0], os.size());

        hpx::serialization::input_archive iarchive(buffer);
        std::vector<T> is; is.resize(os.size());
        iarchive >> hpx::serialization::make_array(&is[0], is.size());
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            HPX_TEST_EQ(os[i], is[i]);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::vector<A<T> > os;
        for(T c = minval; c < maxval; ++c)
        {
            os.push_back(c);
        }
        oarchive << hpx::serialization::make_array(&os[0], os.size());

        hpx::serialization::input_archive iarchive(buffer);
        std::vector<A<T> > is; is.resize(os.size());
        iarchive >> hpx::serialization::make_array(&is[0], is.size());
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            HPX_TEST_EQ(os[i].t_, is[i].t_);
        }
    }
}

template <typename T>
void test_fp(T minval, T maxval)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::vector<T> os;
        for(T c = minval; c < maxval; c += static_cast<T>(0.5))
        {
            os.push_back(c);
        }
        oarchive << hpx::serialization::make_array(&os[0], os.size());

        hpx::serialization::input_archive iarchive(buffer);
        std::vector<T> is; is.resize(os.size());
        iarchive >> hpx::serialization::make_array(&is[0], is.size());
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            HPX_TEST_EQ(os[i], is[i]);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::vector<A<T> > os;
        for(T c = minval; c < maxval; c += static_cast<T>(0.5))
        {
            os.push_back(c);
        }
        oarchive << hpx::serialization::make_array(&os[0], os.size());

        hpx::serialization::input_archive iarchive(buffer);
        std::vector<A<T> > is; is.resize(os.size());
        iarchive >> hpx::serialization::make_array(&is[0], is.size());
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            HPX_TEST_EQ(os[i].t_, is[i].t_);
        }
    }
}

template <class T, std::size_t N>
void test_boost_array(T first)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        boost::array<T, N> oarray;
        std::iota(oarray.begin(), oarray.end(), first);
        oarchive << oarray;

        hpx::serialization::input_archive iarchive(buffer);
        boost::array<T, N> iarray;
        iarchive >> iarray;
        for(std::size_t i = 0; i < oarray.size(); ++i)
        {
            HPX_TEST_EQ(oarray[i], iarray[i]);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        boost::array<A<T>, N> oarray;
        std::iota(oarray.begin(), oarray.end(), first);
        oarchive << oarray;

        hpx::serialization::input_archive iarchive(buffer);
        boost::array<A<T> , N> iarray;
        iarchive >> iarray;
        for(std::size_t i = 0; i < oarray.size(); ++i)
        {
            HPX_TEST_EQ(oarray[i].t_, iarray[i].t_);
        }
    }
}

#ifdef HPX_HAVE_CXX11_STD_ARRAY
template <class T, std::size_t N>
void test_std_array(T first)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::array<T, N> oarray;
        std::iota(oarray.begin(), oarray.end(), first);
        oarchive << oarray;

        hpx::serialization::input_archive iarchive(buffer);
        std::array<T, N> iarray;
        iarchive >> iarray;
        for(std::size_t i = 0; i < oarray.size(); ++i)
        {
            HPX_TEST_EQ(oarray[i], iarray[i]);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer,
            hpx::serialization::disable_data_chunking);
        std::array<A<T>, N> oarray;
        std::iota(oarray.begin(), oarray.end(), first);
        oarchive << oarray;

        hpx::serialization::input_archive iarchive(buffer);
        std::array<A<T> , N> iarray;
        iarchive >> iarray;
        for(std::size_t i = 0; i < oarray.size(); ++i)
        {
            HPX_TEST_EQ(oarray[i].t_, iarray[i].t_);
        }
    }
}
#endif

template <class T>
void test_multi_array(T first)
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer,
        hpx::serialization::disable_data_chunking);
    boost::multi_array<T, 3u> oarray(boost::extents[3][4][2]);

    for(std::size_t i = 0; i < 3; ++i)
        for(std::size_t j = 0; j < 4; ++j)
            for(std::size_t k = 0; k < 2; ++first, ++k)
                oarray[i][j][k] = first;
    oarchive << oarray;

    hpx::serialization::input_archive iarchive(buffer);
    boost::multi_array<T, 3> iarray;
    iarchive >> iarray;
    for(std::size_t i = 0; i < 3; ++i)
        for(std::size_t j = 0; j < 4; ++j)
            for(std::size_t k = 0; k < 2; ++k)
                HPX_TEST_EQ(oarray[i][j][k], iarray[i][j][k]);
}

template <typename T, std::size_t N>
void test_plain_array()
{
    std::vector<char> buffer;
    T iarray[N];
    T oarray[N];

    for(std::size_t i = 0; i < N; ++i) {
        iarray[i] = i * i;
        oarray[i] = -1;
    }

    hpx::serialization::output_archive oarchive(buffer,
        hpx::serialization::disable_data_chunking);
    oarchive << iarray;

    hpx::serialization::input_archive iarchive(buffer);
    iarchive >> oarray;

    for(std::size_t i = 0; i < N; ++i) {
        HPX_TEST_EQ(oarray[i], iarray[i]);
    }
}

template <typename T, std::size_t N>
void test_array_of_vectors()
{
    std::vector<char> buffer;
    std::vector<T> iarray[N];
    std::vector<T> oarray[N];

    for(std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            iarray[i].push_back(i * i);
        }
    }

    hpx::serialization::output_archive oarchive(buffer,
        hpx::serialization::disable_data_chunking);
    oarchive << iarray;

    hpx::serialization::input_archive iarchive(buffer);
    iarchive >> oarray;

    for(std::size_t i = 0; i < N; ++i) {
        HPX_TEST_EQ(oarray[i].size(), iarray[i].size());

        for (std::size_t j = 0; j < i; ++j) {
            HPX_TEST_EQ(oarray[i][j], iarray[i][j]);
        }
    }
}


int main()
{
    test<char>((std::numeric_limits<char>::min)(),
        (std::numeric_limits<char>::max)());
    test<int>((std::numeric_limits<int>::min)(),
        (std::numeric_limits<int>::min)() + 100);
    test<int>((std::numeric_limits<int>::max)() - 100,
        (std::numeric_limits<int>::max)());
    test<int>(-100, 100);
    test<unsigned>((std::numeric_limits<unsigned>::min)(),
        (std::numeric_limits<unsigned>::min)() + 100);
    test<unsigned>((std::numeric_limits<unsigned>::max)() - 100,
        (std::numeric_limits<unsigned>::max)());
    test<long>((std::numeric_limits<long>::min)(),
        (std::numeric_limits<long>::min)() + 100);
    test<long>((std::numeric_limits<long>::max)() - 100,
        (std::numeric_limits<long>::max)());
    test<long>(-100, 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::min)(),
        (std::numeric_limits<unsigned long>::min)() + 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::max)() - 100,
        (std::numeric_limits<unsigned long>::max)());
    test_fp<float>((std::numeric_limits<float>::min)(),
        (std::numeric_limits<float>::min)() + 100);
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>(-100, 100);

    test_boost_array<char, 100U>('\0');
    test_boost_array<double, 40U>((std::numeric_limits<double>::min)());
    test_boost_array<float, 100U>(0.f);

#ifdef HPX_HAVE_CXX11_STD_ARRAY
    test_std_array<char, 100U>('\0');
    test_std_array<double, 40U>((std::numeric_limits<double>::min)());
    test_std_array<float, 100U>(0.f);
#endif

    test_multi_array(0);
    test_multi_array(0.);

    test_plain_array<double, 20>();
    test_plain_array<int, 200>();

    test_array_of_vectors<double, 20>();
    test_array_of_vectors<int, 200>();

    return hpx::util::report_errors();
}

#endif
