//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>
#include <hpx/runtime/serialization/intrusive_ptr.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <memory>
#include <vector>

void test_boost_shared()
{
    boost::shared_ptr<int> ip(new int(7));
    boost::shared_ptr<int> op1;
    boost::shared_ptr<int> op2;
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
    HPX_TEST_EQ(*op1, *ip);
    op1.reset();
    HPX_TEST_EQ(*op2, *ip);
}

void test_shared()
{
    std::shared_ptr<int> ip(new int(7));
    std::shared_ptr<int> op1;
    std::shared_ptr<int> op2;
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
    HPX_TEST_EQ(*op1, *ip);
    op1.reset();
    HPX_TEST_EQ(*op2, *ip);
}

void test_unique()
{
    std::unique_ptr<int> ip(new int(7));
    std::unique_ptr<int> op1;
    std::unique_ptr<int> op2;
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
    HPX_TEST_NEQ(op1.get(), op2.get()); //untracked
    HPX_TEST_EQ(*op1, *ip);
    HPX_TEST_EQ(*op2, *ip);
}

struct A
{
    A() : i(7), count(0) {}
    int i;

    int count;

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & i;
    }
};

void intrusive_ptr_add_ref(A* a)
{
    ++a->count;
}

void intrusive_ptr_release(A* a)
{
    if(--a->count == 0)
    {
        delete a;
    }
}

void test_intrusive()
{
    boost::intrusive_ptr<A> ip(new A());
    boost::intrusive_ptr<A> op1;
    boost::intrusive_ptr<A> op2;
    {
        std::vector<char> buffer;
        hpx::serialization::output_archive oarchive(buffer);
        oarchive << ip << ip;

        hpx::serialization::input_archive iarchive(buffer);
        iarchive >> op1;
        iarchive >> op2;
    }
    HPX_TEST_EQ(ip->count, 1);
    HPX_TEST_NEQ(op1.get(), ip.get());
    HPX_TEST_NEQ(op2.get(), ip.get());
    HPX_TEST_EQ(op1.get(), op2.get());
    HPX_TEST_EQ(op1->i, ip->i);
    HPX_TEST_EQ(op1->count, 2);
    HPX_TEST_EQ(op2->count, 2);
    op1.reset();
    HPX_TEST_EQ(op2->count, 1);
    HPX_TEST_EQ(op2->i, ip->i);
}

int main()
{
    test_boost_shared();
    test_shared();
    test_unique();
    test_intrusive();

    return hpx::util::report_errors();
}
