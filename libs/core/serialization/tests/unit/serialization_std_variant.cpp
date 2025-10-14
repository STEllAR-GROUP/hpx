//  Copyright (c) 2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/serialization/variant.hpp>

#include <hpx/modules/testing.hpp>

#include <string>
#include <variant>
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
        ar & t_;
    }
};

int main()
{
    std::vector<char> buf;
    hpx::serialization::output_archive oar(buf);
    hpx::serialization::input_archive iar(buf);

    std::variant<int, std::string, double, A<int>> ovar = std::string("dfsdf");
    std::variant<int, std::string, double, A<int>> ivar;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.index(), ovar.index());
    HPX_TEST(ivar == ovar);

    ovar = 2.5;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.index(), ovar.index());
    HPX_TEST(ivar == ovar);

    ovar = 1;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.index(), ovar.index());
    HPX_TEST(ivar == ovar);

    ovar = A<int>(2);
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.index(), ovar.index());
    HPX_TEST(ivar == ovar);

    const std::variant<std::string> sovar = std::string("string");
    std::variant<std::string> sivar;
    oar << sovar;
    iar >> sivar;

    HPX_TEST_EQ(sivar.index(), sovar.index());
    HPX_TEST(sivar == sovar);

    bool caught_exception = false;
    try
    {
        std::variant<std::string, int> sovar = 42;
        std::variant<std::string> sivar;
        oar << sovar;
        iar >> sivar;

        HPX_TEST(false);
    }
    catch (hpx::exception const& e)
    {
        if (e.get_error() == hpx::error::serialization_error)
        {
            caught_exception = true;
        }
    }
    HPX_TEST(caught_exception);

    return hpx::util::report_errors();
}
