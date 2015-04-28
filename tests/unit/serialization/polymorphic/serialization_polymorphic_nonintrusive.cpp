//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/base_object.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

struct A
{
    A() : a(8) {}
    virtual ~A() {}

    int a;
};

template <typename Archive>
void serialize(Archive& ar, A& a, unsigned)
{
  ar & a.a;
}

HPX_SERIALIZATION_REGISTER_CLASS(A);

struct B
{
    B() : b(6) {}
    explicit B(int i) : b(i) {}

    virtual ~B() {}

    virtual void f() = 0;

    int b;

};

template <class Archive>
void serialize(Archive& ar, B& b, unsigned)
{
  ar & b.b;
}

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(B);

struct D : B
{
    D() : d(89) {}
    explicit D(int i) : B(i), d(89) {}
    void f() {}

    int d;
};

template <class Archive>
void serialize(Archive& ar, D& d, unsigned)
{
  d.b = 4711;
  ar & hpx::serialization::base_object<B>(d);
  ar & d.d;
}

HPX_SERIALIZATION_REGISTER_CLASS(D);

int main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);
    oarchive << A();
    D d;
    B const & b1 = d;
    oarchive << b1;

    hpx::serialization::input_archive iarchive(buffer);
    A a;
    iarchive >> a;
    D d1;
    B & b2 = d1;
    iarchive >> b2;
    HPX_TEST_EQ(a.a, 8);
    HPX_TEST_EQ(&b2, &d1);
    HPX_TEST_EQ(b2.b, d1.b);
    HPX_TEST_EQ(d.b, d1.b);
    HPX_TEST_EQ(d.d, d1.d);

    return hpx::util::report_errors();
}
