//
// ip_hash_test.cpp
//
// Copyright 2011 Peter Dimov
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt
//

#include <hpx/modules/memory.hpp>
#include <hpx/modules/testing.hpp>

#include <functional>

class base
{
private:
    int use_count_;

    base(base const&) = delete;
    base& operator=(base const&) = delete;

protected:
    base()
      : use_count_(0)
    {
    }

    virtual ~base() {}

public:
    long use_count() const
    {
        return use_count_;
    }

    inline friend void intrusive_ptr_add_ref(base* p)
    {
        ++p->use_count_;
    }

    inline friend void intrusive_ptr_release(base* p)
    {
        if (--p->use_count_ == 0)
            delete p;
    }
};

struct X : public base
{
};

int main()
{
    std::hash<hpx::intrusive_ptr<X>> hasher;

    hpx::intrusive_ptr<X> p1, p2(p1), p3(new X), p4(p3), p5(new X);

    HPX_TEST_EQ(p1, p2);
    HPX_TEST_EQ(hasher(p1), hasher(p2));

    HPX_TEST_NEQ(p1, p3);
    HPX_TEST_NEQ(hasher(p1), hasher(p3));

    HPX_TEST_EQ(p3, p4);
    HPX_TEST_EQ(hasher(p3), hasher(p4));

    HPX_TEST_NEQ(p3, p5);
    HPX_TEST_NEQ(hasher(p3), hasher(p5));

    return hpx::util::report_errors();
}
