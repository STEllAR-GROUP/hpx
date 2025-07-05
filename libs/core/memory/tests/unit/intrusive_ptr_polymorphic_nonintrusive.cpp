//  Copyright (c) 2014-2015 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/memory/serialization/intrusive_ptr.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

// =========================intrusive_ptr test==============================
struct D
{
    int a;
    int count;

    explicit D(int a = 1)
      : a(a)
      , count(0)
    {
    }
    virtual ~D() {}

    virtual const char* foo() = 0;
};
HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(D)

template <class Archive>
void load(Archive& ar, D& d, unsigned)
{
    ar & d.a;
}
template <class Archive>
void save(Archive& ar, const D& d, unsigned)
{
    ar & d.a;
}
HPX_SERIALIZATION_SPLIT_FREE(D)

void intrusive_ptr_add_ref(D* d) noexcept
{
    ++d->count;
}

void intrusive_ptr_release(D* d) noexcept
{
    if (--d->count == 0)
    {
        delete d;
    }
}

struct E : D
{
    int b;

    explicit E(int b = 2)
      : b(b)
    {
    }

    const char* foo() override
    {
        return "E::foo";
    }
};

template <class Archive>
void load(Archive& ar, E& e, unsigned)
{
    ar& hpx::serialization::base_object<D>(e);
    ar & e.b;
}
template <class Archive>
void save(Archive& ar, const E& e, unsigned)
{
    ar& hpx::serialization::base_object<D>(e);
    ar & e.b;
}
HPX_SERIALIZATION_SPLIT_FREE(E)
HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(E)

struct F : public E
{
    int c;

    explicit F(int c = 3)
      : c(c)
    {
    }

    virtual const char* foo()
    {
        return "F::foo";
    }

    int get_c() const
    {
        return c;
    }
};

template <class Archive>
void serialize(Archive& ar, F& f, unsigned)
{
    ar& hpx::serialization::base_object<E>(f);
    ar & f.c;
}
HPX_SERIALIZATION_REGISTER_CLASS(F)

void test_intrusive()
{
    hpx::intrusive_ptr<D> ip(new F);
    hpx::intrusive_ptr<D> op1;
    hpx::intrusive_ptr<D> op2;

    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }

    HPX_TEST_NEQ(op1.get(), ip.get());
    HPX_TEST_NEQ(op2.get(), ip.get());
    HPX_TEST_EQ(op1.get(), op2.get());
    HPX_TEST_EQ(op1->foo(), std::string("F::foo"));
    HPX_TEST_EQ(op2->foo(), std::string("F::foo"));
    HPX_TEST_EQ(static_cast<F*>(op1.get())->a, 1);
    HPX_TEST_EQ(static_cast<F*>(op1.get())->b, 2);
    HPX_TEST_EQ(static_cast<F*>(op1.get())->get_c(), 3);
    HPX_TEST_EQ(static_cast<F*>(op2.get())->a, 1);
    HPX_TEST_EQ(static_cast<F*>(op2.get())->b, 2);
    HPX_TEST_EQ(static_cast<F*>(op2.get())->get_c(), 3);

    HPX_TEST_EQ(ip->count, 1);
    HPX_TEST_EQ(op1->count, 2);
    HPX_TEST_EQ(op2->count, 2);
    op1.reset();
    HPX_TEST_EQ(op2->count, 1);
}

int main()
{
    test_intrusive();

    return hpx::util::report_errors();
}
