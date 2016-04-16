//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

struct A
{
    A() : a(8) {}
    virtual ~A() {}

    int a;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & a;
    }
    HPX_SERIALIZATION_POLYMORPHIC(A);
};

struct B
{
    B() : b(6) {}
    explicit B(int i) : b(i) {}

    virtual ~B() {}

    virtual void f() = 0;

    int b;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & b;
    }
    HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(B);
};

struct D : B
{
    D() : d(89) {}
    explicit D(int i) : B(i), d(89) {}
    void f() {}

    int d;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        b = 4711;
        ar & hpx::serialization::base_object<B>(*this);;
        ar & d;
    }
    HPX_SERIALIZATION_POLYMORPHIC(D);
};

int main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);
    oarchive << A();

    B * const b1 = new D;
    oarchive << hpx::serialization::detail::raw_ptr(b1);
    oarchive << hpx::serialization::detail::raw_ptr(b1);

    hpx::serialization::input_archive iarchive(buffer);
    A a;
    iarchive >> a;
    B *b2 = 0, *b3 = 0;
    iarchive >> hpx::serialization::detail::raw_ptr(b2);
    iarchive >> hpx::serialization::detail::raw_ptr(b3);

    HPX_TEST_EQ(a.a, 8);
    HPX_TEST_NEQ(b2, b1);
    HPX_TEST_NEQ(b2, b3); //untracked
    HPX_TEST_EQ(b2->b, b1->b);

    delete b2;
    delete b3;

    HPX_TEST_EQ(b1->b, 4711);

    delete b1;

    return hpx::util::report_errors();
}
