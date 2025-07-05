//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstdint>
#include <vector>

template <typename T>
struct A
{
    A() {}

    explicit A(T t)
      : t_(t)
    {
    }
    T t_;

    A& operator=(T t)
    {
        t_ = t;
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & t_;
    }
};

#include <hpx/modules/testing.hpp>

void test_bool()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        bool b = true;
        oarchive << b;
        b = false;
        oarchive << b;
        b = false;
        oarchive << b;
        b = true;
        oarchive << b;

        hpx::serialization::input_archive iarchive(buffer);
        b = false;
        iarchive >> b;
        HPX_TEST_EQ(b, true);
        iarchive >> b;
        HPX_TEST_EQ(b, false);
        iarchive >> b;
        HPX_TEST_EQ(b, false);
        iarchive >> b;
        HPX_TEST_EQ(b, true);
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        auto b = A<bool>(true);
        oarchive << b;
        b = false;
        oarchive << b;
        b = false;
        oarchive << b;
        b = true;
        oarchive << b;

        hpx::serialization::input_archive iarchive(buffer);
        b = false;
        iarchive >> b;
        HPX_TEST_EQ(b.t_, true);
        iarchive >> b;
        HPX_TEST_EQ(b.t_, false);
        iarchive >> b;
        HPX_TEST_EQ(b.t_, false);
        iarchive >> b;
        HPX_TEST_EQ(b.t_, true);
    }
}

template <typename T>
void test(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        for (T c = min; c < max; ++c)
        {
            oarchive << c;
        }
        hpx::serialization::input_archive iarchive(buffer);
        for (T c = min; c < max; ++c)
        {
            T cc = 0;
            iarchive >> cc;
            HPX_TEST(c == cc);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        for (T c = min; c < max; ++c)
        {
            auto cc = A<T>(c);
            oarchive << cc;
        }
        hpx::serialization::input_archive iarchive(buffer);
        for (T c = min; c < max; ++c)
        {
            auto cc = A<T>(0);
            iarchive >> cc;
            HPX_TEST(c == cc.t_);
        }
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            oarchive << c;
        }
        hpx::serialization::input_archive iarchive(buffer);
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            T cc = 0;
            iarchive >> cc;
            HPX_TEST(c == cc);
        }
    }
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            auto cc = A<T>(c);
            oarchive << cc;
        }
        hpx::serialization::input_archive iarchive(buffer);
        for (T c = min; c < max; c += static_cast<T>(0.5))
        {
            auto cc = A<T>(0);
            iarchive >> cc;
            HPX_TEST(c == cc.t_);
        }
    }
}

int main()
{
    test_bool();
    test<char>(
        (std::numeric_limits<char>::min)(), (std::numeric_limits<char>::max)());
    test<signed char>((std::numeric_limits<signed char>::min)(),
        (std::numeric_limits<signed char>::max)());
    test<unsigned char>((std::numeric_limits<unsigned char>::min)(),
        (std::numeric_limits<unsigned char>::max)());
    test<short>((std::numeric_limits<short>::min)(),
        (std::numeric_limits<short>::min)() + 100);
    test<short>((std::numeric_limits<short>::max)() - 100,
        (std::numeric_limits<short>::max)());
    test<short>(-100, 100);
    test<unsigned short>((std::numeric_limits<unsigned short>::min)(),
        (std::numeric_limits<unsigned short>::min)() + 100);
    test<unsigned short>((std::numeric_limits<unsigned short>::max)() - 100,
        (std::numeric_limits<unsigned short>::max)());
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
    test<long long>((std::numeric_limits<long long>::min)(),
        (std::numeric_limits<long long>::min)() + 100);
    test<long long>((std::numeric_limits<long long>::max)() - 100,
        (std::numeric_limits<long long>::max)());
    test<long long>(-100, 100);
    test<unsigned long long>((std::numeric_limits<unsigned long long>::min)(),
        (std::numeric_limits<unsigned long long>::min)() + 100);
    test<unsigned long long>(
        (std::numeric_limits<unsigned long long>::max)() - 100,
        (std::numeric_limits<unsigned long long>::max)());
    test_fp<float>((std::numeric_limits<float>::min)(),
        (std::numeric_limits<float>::min)() + 100);
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>(-100, 100);
    test<long double>((std::numeric_limits<long double>::min)(),
        (std::numeric_limits<long double>::min)() + 100);
    test<long double>(-100, 100);

    return hpx::util::report_errors();
}
