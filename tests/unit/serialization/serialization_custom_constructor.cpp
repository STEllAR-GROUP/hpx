//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

/**
 * In this class we use the custom c-tor macro to simply delegate to
 * another c-tor without actually accessing the archive.
 */
struct A
{
    A(int a) : a(a) {}
    virtual ~A() {}

    int a;
};

template <typename Archive>
void serialize(Archive& ar, A& a, unsigned)
{
    ar & a.a;
}

A *a_factory(hpx::serialization::input_archive& ar)
{
    A *a = new A(123);
    ar >> *a;
    return a;
}

HPX_SERIALIZATION_REGISTER_CLASS(A);
HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(A, a_factory);

/**
 * In this example we leverage the custom factory to pull data from
 * the archive and prevent duplicate construction of potentially
 * expensive members. Side effect: serialize() needs to be specialized
 * for input archives as to avoid duplicate reads.
 */
struct B
{
    B(double b, bool flag) :
        b(b)
    {
        if (flag) {
            std::cout << "B(" << b << ")\n";
        }
    }

    virtual ~B() {}

    double b;
};

template <typename Archive>
void serialize(Archive& ar, B& b, unsigned)
{
    ar & b.b;
}

B *b_factory(hpx::serialization::input_archive& ar)
{
    double b;
    ar & b;

    bool flag = (b < 8);
    return new B(b, flag);
}

HPX_SERIALIZATION_REGISTER_CLASS(B);
HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(B, b_factory);

/**
 * Obviously we need to check templates, too.
 */
template<typename T>
struct C
{
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
C<T> *c_factory(hpx::serialization::input_archive& ar, C<T> */*unused*/)
{
    C<T> *c = new C<T>(666);
    ar >> *c;
    return c;
}

HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE((template<typename T>),
    (C<T>), c_factory);

void test_delegate()
{
    std::vector<char> buffer;
    {
        boost::shared_ptr<A> struct_a(new A(4711));
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << struct_a;
    }
    {
        boost::shared_ptr<A> struct_b;
        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> struct_b;
        HPX_TEST_EQ(struct_b->a, 4711);
    }
}

void test_custom_factory()
{
    std::vector<char> buffer;

    {
        boost::shared_ptr<B> struct_a(new B(1981, false));
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << struct_a;
    }
    {
        boost::shared_ptr<B> struct_b;
        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> struct_b;
        HPX_TEST_EQ(struct_b->b, 1981);
    }
}

void test_template()
{
    std::vector<char> buffer;
    {
        boost::shared_ptr<C<float> > struct_a(new C<float>(777));
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << struct_a;
    }
    {
        boost::shared_ptr<C<float> > struct_b;
        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> struct_b;
        HPX_TEST_EQ(struct_b->c, 777);
    }
}

int main()
{
    test_delegate();
    test_custom_factory();
    test_template();

    return hpx::util::report_errors();
}
