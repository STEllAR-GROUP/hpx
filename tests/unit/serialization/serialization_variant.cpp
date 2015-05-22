//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/runtime/serialization/variant.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

template <typename T>
struct A
{
    A() {}

    A(T t) : t_(t) {}
    T t_;

    A & operator=(T t) { t_ = t; return *this; }

    friend bool operator==(A a, A b)
    {
        return a.t_ == b.t_;
    }

    friend std::ostream& operator<<(std::ostream& os, A a)
    {
        os << a;
        return os;
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & t_;
    }
};

int main()
{
    std::vector<char> buf;
    hpx::serialization::output_archive oar(buf);
    hpx::serialization::input_archive iar(buf);

    boost::variant<int, std::string, double, A<int> > ovar = std::string("dfsdf");
    boost::variant<int, std::string, double, A<int> > ivar;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.which(), ovar.which());
    HPX_TEST_EQ(ivar, ovar);

    ovar = 2.5;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.which(), ovar.which());
    HPX_TEST_EQ(ivar, ovar);

    ovar = 1;
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.which(), ovar.which());
    HPX_TEST_EQ(ivar, ovar);

    ovar = A<int>(2);
    oar << ovar;
    iar >> ivar;

    HPX_TEST_EQ(ivar.which(), ovar.which());
    HPX_TEST_EQ(ivar, ovar);

    const boost::variant<std::string> sovar = std::string("string");
    boost::variant<std::string> sivar;
    oar << sovar;
    iar >> sivar;

    HPX_TEST_EQ(sivar.which(), sovar.which());
    HPX_TEST_EQ(sivar, sovar);

    return hpx::util::report_errors();
}
