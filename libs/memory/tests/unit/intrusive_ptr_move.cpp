//
//  intrusive_ptr_move_test.cpp
//
//  Copyright (c) 2002-2005 Peter Dimov
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>

#if defined(HPX_MSVC)

#pragma warning(disable : 4786)    // identifier truncated in debug info
#pragma warning(disable : 4710)    // function not inlined
#pragma warning(                                                               \
    disable : 4711)    // function selected for automatic inline expansion
#pragma warning(disable : 4514)    // unreferenced inline removed
#pragma warning(                                                               \
    disable : 4355)    // 'this' : used in base member initializer list
#pragma warning(disable : 4511)    // copy constructor could not be generated
#pragma warning(disable : 4512)    // assignment operator could not be generated
#pragma warning(disable : 4675)    // resolved overload found with Koenig lookup

#endif

#include <hpx/modules/memory.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <utility>

namespace N {

    class base
    {
    private:
        mutable hpx::util::atomic_count use_count_;

        base(base const&) = delete;
        base& operator=(base const&) = delete;

    protected:
        base()
          : use_count_(0)
        {
            ++instances;
        }

        virtual ~base()
        {
            --instances;
        }

    public:
        static long instances;

        long use_count() const
        {
            return use_count_;
        }

        inline friend void intrusive_ptr_add_ref(base const* p)
        {
            ++p->use_count_;
        }

        inline friend void intrusive_ptr_release(base const* p)
        {
            if (--p->use_count_ == 0)
                delete p;
        }
    };

    long base::instances = 0;

}    // namespace N

//
struct X : public virtual N::base
{
};

struct Y : public X
{
};

int main()
{
    HPX_TEST_EQ(N::base::instances, 0);

    {
        hpx::intrusive_ptr<X> p(new X);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2(std::move(p));
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<Y> p(new Y);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2(std::move(p));
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<X> p(new X);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2;
        p2 = std::move(p);
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<X> p(new X);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2(new X);
        HPX_TEST_EQ(N::base::instances, 2);
        p2 = std::move(p);
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<Y> p(new Y);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2;
        p2 = std::move(p);
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<Y> p(new Y);
        HPX_TEST_EQ(N::base::instances, 1);

        hpx::intrusive_ptr<X> p2(new X);
        HPX_TEST_EQ(N::base::instances, 2);
        p2 = std::move(p);
        HPX_TEST_EQ(N::base::instances, 1);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(p.get() == nullptr);

        p2.reset();
        HPX_TEST_EQ(N::base::instances, 0);
    }

    {
        hpx::intrusive_ptr<X> px(new Y);

        X* px2 = px.get();

        hpx::intrusive_ptr<Y> py = hpx::static_pointer_cast<Y>(std::move(px));
        HPX_TEST_EQ(py.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(px.get() == nullptr);
        HPX_TEST_EQ(py->use_count(), 1);
    }

    HPX_TEST_EQ(N::base::instances, 0);

    {
        hpx::intrusive_ptr<X const> px(new X);

        X const* px2 = px.get();

        hpx::intrusive_ptr<X> px3 = hpx::const_pointer_cast<X>(std::move(px));
        HPX_TEST_EQ(px3.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(px.get() == nullptr);
        HPX_TEST_EQ(px3->use_count(), 1);
    }

    HPX_TEST_EQ(N::base::instances, 0);

    {
        hpx::intrusive_ptr<X> px(new Y);

        X* px2 = px.get();

        hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(std::move(px));
        HPX_TEST_EQ(py.get(), px2);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST(px.get() == nullptr);
        HPX_TEST_EQ(py->use_count(), 1);
    }

    HPX_TEST_EQ(N::base::instances, 0);

    {
        hpx::intrusive_ptr<X> px(new X);

        X* px2 = px.get();

        hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(std::move(px));
        HPX_TEST(py.get() == nullptr);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        HPX_TEST_EQ(px.get(), px2);
        HPX_TEST_EQ(px->use_count(), 1);
    }

    HPX_TEST_EQ(N::base::instances, 0);

    return hpx::util::report_errors();
}
