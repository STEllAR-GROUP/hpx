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

A *a_factory()
{
    return new A(123);
}

HPX_SERIALIZATION_REGISTER_CLASS(A);
HPX_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(A, a_factory);
void test_shared()
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

int main()
{
    test_shared();

    return hpx::util::report_errors();
}
