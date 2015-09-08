//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/map.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <iterator>
#include <numeric>

template <typename T>
struct A
{
    A() {}

    A(T t) : t_(t) {}
    T t_;

    A & operator=(T t) { t_ = t; return *this; }
    bool operator==(const A& a) const { return t_ == a.t_; }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & t_;
    }
};

template <class T>
std::ostream& operator<<(std::ostream& os, const A<T>& a)
{
    return os << a.t_;
}

template <typename T>
void test(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::map<T, A<T> > os;
        for(T c = min; c < max; ++c)
        {
            os.insert(std::make_pair(c, A<T>(c)));
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::map<T, A<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (const auto& v: os)
        {
            HPX_TEST_EQ(os[v.first], is[v.first]);
        }
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        std::map<T, A<T> > os;
        for(T c = min; c < max; c += static_cast<T>(0.5))
        {
            os.insert(std::make_pair(c, A<T>(c)));
        }
        oarchive << os;
        hpx::serialization::input_archive iarchive(buffer);
        std::map<T, A<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for (const auto& v: os)
        {
            HPX_TEST_EQ(os[v.first], is[v.first]);
        }
    }
}

// prohibited, but for adl
namespace std
{
    std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec)
    {
        std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(os, " "));
        return os;
    }
}

void test_vector_as_value()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);
    std::map<size_t, std::vector<int> > os;
    for (size_t k = 0; k < 10; ++k)
    {
        std::vector<int> vec(10);
        std::iota(vec.begin(), vec.end(), k);
        os.insert(std::make_pair(k, vec));
    }
    oarchive << os;
    hpx::serialization::input_archive iarchive(buffer);
    std::map<size_t, std::vector<int> > is;
    iarchive >> is;
    HPX_TEST_EQ(os.size(), is.size());
    for (const auto& v: os)
    {
        HPX_TEST_EQ(os[v.first], is[v.first]);
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

    test_vector_as_value();

    return hpx::util::report_errors();
}
