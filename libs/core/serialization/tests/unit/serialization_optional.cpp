//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/datastructures.hpp>
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
        ar& t_;
    }
};

int main()
{
    std::vector<char> buf;
    hpx::serialization::output_archive oar(buf);
    hpx::serialization::input_archive iar(buf);

    {
        hpx::util::optional<int> ovar;
        hpx::util::optional<int> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar == hpx::util::nullopt);
    }

    {
        hpx::util::optional<int> ovar = 42;
        hpx::util::optional<int> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::util::nullopt);
    }

    {
        hpx::util::optional<double> ovar = 2.5;
        hpx::util::optional<double> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::util::nullopt);
    }

    {
        hpx::util::optional<A<int>> ovar = A<int>{2};
        hpx::util::optional<A<int>> ivar;
        oar << ovar;
        iar >> ivar;

        HPX_TEST_EQ(ivar.has_value(), ovar.has_value());
        HPX_TEST(ivar == ovar);
        HPX_TEST(ivar != hpx::util::nullopt);
    }

    return hpx::util::report_errors();
}
