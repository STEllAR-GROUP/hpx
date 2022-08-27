//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/datastructures/optional.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/optional.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>

#include <hpx/modules/testing.hpp>

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

    friend bool operator==(A a, A b)
    {
        return a.t_ == b.t_;
    }

    friend std::ostream& operator<<(std::ostream& os, A a)
    {
        os << a.t_;
        return os;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & t_;
        // clang-format on
    }
};

int main()
{
    std::vector<char> buf;
    hpx::serialization::output_archive oar(buf);
    hpx::serialization::input_archive iar(buf);

    {
        hpx::optional<int> ovar;
        hpx::optional<int> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar == hpx::nullopt);
    }

    {
        hpx::optional<int> ovar(42);
        hpx::optional<int> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::nullopt);
    }

    {
        hpx::optional<double> ovar(2.5);
        hpx::optional<double> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::nullopt);
    }

    {
        hpx::optional<A<int>> ovar(A<int>{2});
        hpx::optional<A<int>> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::nullopt);
    }

    return hpx::util::report_errors();
}
