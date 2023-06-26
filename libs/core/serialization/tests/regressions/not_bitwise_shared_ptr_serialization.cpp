//  Copyright (c) 2021-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// enforce that types are bitwise serializable by default
#define HPX_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE

// enforce pointers being serializable
#define HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION

#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <memory>
#include <vector>

struct A
{
    A() = default;

    A(int a, double b)
      : a(a)
      , b(b)
    {
    }

    int a = 0;
    double b = 0.0;
    bool serialized = false;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & a & b;
        // clang-format on

        serialized = true;
    }
};

int hpx_main()
{
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::shared_ptr<A> oa(new A(42, 42.0));
        std::shared_ptr<A> ob(oa);

        oarchive << oa << ob;
        HPX_TEST(oa->serialized);

        hpx::serialization::input_archive iarchive(buffer);
        std::shared_ptr<A> ia;
        std::shared_ptr<A> ib;

        iarchive >> ia >> ib;
        HPX_TEST(ia->serialized);

        HPX_TEST(ia.get() == ib.get());

        HPX_TEST_EQ(ia->a, 42);
        HPX_TEST_EQ(ia->b, 42.0);
    }

    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);

        std::shared_ptr<A> oa = std::make_shared<A>(42, 42.0);
        std::shared_ptr<A> ob(oa);

        oarchive << oa << ob;
        HPX_TEST(oa->serialized);

        hpx::serialization::input_archive iarchive(buffer);
        std::shared_ptr<A> ia;
        std::shared_ptr<A> ib;

        iarchive >> ia >> ib;
        HPX_TEST(ia->serialized);

        HPX_TEST(ia.get() == ib.get());

        HPX_TEST_EQ(ia->a, 42);
        HPX_TEST_EQ(ia->b, 42.0);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
