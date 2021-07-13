//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// enforce that types are bitwise serializable by default
#define HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE

#include <hpx/config.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/is_not_bitwise_serializable.hpp>

#include <vector>

bool serialize_A = false;
bool serialize_B = false;

struct A
{
    int a;
    double b;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & a & b;
        // clang-format on

        serialize_A = true;
    }
};

struct A_nonser
{
    int a;
    double b;
};

struct B
{
    int a;
    double b;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & a & b;
        // clang-format on

        serialize_B = true;
    }
};

HPX_IS_NOT_BITWISE_SERIALIZABLE(B);

int hpx_main()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        serialize_A = false;
        A oa{42, 42.0};
        oarchive << oa;
        HPX_TEST(serialize_A);

        hpx::serialization::input_archive iarchive(buffer);
        A ia{0, 0.0};

        serialize_A = false;
        iarchive >> ia;
        HPX_TEST(serialize_A);

        HPX_TEST_EQ(ia.a, 42);
        HPX_TEST_EQ(ia.b, 42.0);
    }

    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        A_nonser oa{42, 42.0};
        oarchive << oa;

        hpx::serialization::input_archive iarchive(buffer);
        A_nonser ia{0, 0.0};

        iarchive >> ia;

        HPX_TEST_EQ(ia.a, 42);
        HPX_TEST_EQ(ia.b, 42.0);
    }

    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        serialize_B = false;
        B ob{42, 42.0};
        oarchive << ob;
        HPX_TEST(serialize_B);

        hpx::serialization::input_archive iarchive(buffer);
        B ib{0, 0.0};

        serialize_B = false;
        iarchive >> ib;
        HPX_TEST(serialize_B);

        HPX_TEST_EQ(ib.a, 42);
        HPX_TEST_EQ(ib.b, 42.0);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
