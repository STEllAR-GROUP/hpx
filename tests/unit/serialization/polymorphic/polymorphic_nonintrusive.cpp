//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/base_object.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <memory>
#include <vector>

struct A
{
    A(int a = 8) : a(a) {}
    virtual ~A() {}

    int a;
};

template <typename Archive>
void serialize(Archive& ar, A& a, unsigned)
{
  ar & a.a;
}

HPX_SERIALIZATION_REGISTER_CLASS(A);
HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(A);

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

template<typename T>
struct C
{
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(C);

    C(T c) :
        c(c)
    {}

    T c;
};

template<typename Archive, typename T>
void serialize(Archive& ar, C<T>& c, unsigned)
{
    ar & c.c;
}

template<typename T>
C<T> *c_factory(hpx::serialization::input_archive& ar, C<T> * /*unused*/)
{
    C<T> *c = new C<T>(999);
    serialize(ar, *c, 0);
    return c;
}

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), C<T>)
HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template<class T>, C<T>)
HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE((template<typename T>),
    (C<T>), c_factory);

template<typename T>
struct E : public A
{
public:
    HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(E);

    E(int i, T t) :
        A(i),
        c(t)
    {}

    C<T> c;
};

namespace hpx { namespace serialization {

    template <class Archive, class T>
    void serialize(Archive& archive, E<T>& s, unsigned)
    {
        archive & hpx::serialization::base_object<A>(s);
        archive & s.c;
    }
} }

template<typename T>
E<T> *e_factory(hpx::serialization::input_archive& ar, E<T> * /*unused*/)
{
    E<T> *e = new E<T>(99, 9999);
    serialize(ar, *e, 0);
    return e;
}

HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE((template <class T>), E<T>)
HPX_SERIALIZATION_REGISTER_CLASS_TEMPLATE(template<class T>, E<T>)
HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE((template<typename T>),
    (E<T>), e_factory);

void test_basic()
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
}

void test_member()
{
    std::vector<char> buffer;
    {
        std::shared_ptr<A> struct_a(new E<float>(1, 2.3f));
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << struct_a;
    }
    {
        std::shared_ptr<A> struct_b;
        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> struct_b;
        HPX_TEST_EQ(struct_b->a, 1);
        HPX_TEST_EQ(dynamic_cast<E<float>*>(&*struct_b)->c.c, 2.3f);
    }
}


int main()
{
    test_basic();
    test_member();

    return hpx::util::report_errors();
}
