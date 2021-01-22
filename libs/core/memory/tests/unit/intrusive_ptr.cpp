//
//  intrusive_ptr_test.cpp
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

#include <algorithm>
#include <functional>
#include <utility>

//
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

//
namespace n_element_type {

    void f(X&) {}

    void test()
    {
        using T = hpx::intrusive_ptr<X>::element_type;

        T t;
        f(t);
    }

}    // namespace n_element_type

namespace n_constructors {

    void default_constructor()
    {
        hpx::intrusive_ptr<X> px;
        HPX_TEST(px.get() == nullptr);
    }

    void pointer_constructor()
    {
        {
            hpx::intrusive_ptr<X> px(nullptr);
            HPX_TEST(px.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<X> px(nullptr, false);
            HPX_TEST(px.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 1);

            hpx::intrusive_ptr<X> px(p);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 1);

            intrusive_ptr_add_ref(p);
            HPX_TEST_EQ(p->use_count(), 1);

            hpx::intrusive_ptr<X> px(p, false);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

    void copy_constructor()
    {
        {
            hpx::intrusive_ptr<X> px;
            hpx::intrusive_ptr<X> px2(px);
            HPX_TEST_EQ(px2.get(), px.get());
        }

        {
            hpx::intrusive_ptr<Y> py;
            hpx::intrusive_ptr<X> px(py);
            HPX_TEST_EQ(px.get(), py.get());
        }

        {
            hpx::intrusive_ptr<X> px(nullptr);
            hpx::intrusive_ptr<X> px2(px);
            HPX_TEST_EQ(px2.get(), px.get());
        }

        {
            hpx::intrusive_ptr<Y> py(nullptr);
            hpx::intrusive_ptr<X> px(py);
            HPX_TEST_EQ(px.get(), py.get());
        }

        {
            hpx::intrusive_ptr<X> px(nullptr, false);
            hpx::intrusive_ptr<X> px2(px);
            HPX_TEST_EQ(px2.get(), px.get());
        }

        {
            hpx::intrusive_ptr<Y> py(nullptr, false);
            hpx::intrusive_ptr<X> px(py);
            HPX_TEST_EQ(px.get(), py.get());
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            hpx::intrusive_ptr<X> px2(px);
            HPX_TEST_EQ(px2.get(), px.get());

            HPX_TEST_EQ(N::base::instances, 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<Y> py(new Y);
            hpx::intrusive_ptr<X> px(py);
            HPX_TEST_EQ(px.get(), py.get());

            HPX_TEST_EQ(N::base::instances, 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

    void test()
    {
        default_constructor();
        pointer_constructor();
        copy_constructor();
    }

}    // namespace n_constructors

namespace n_destructor {

    void test()
    {
        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);

            {
                hpx::intrusive_ptr<X> px2(px);
                HPX_TEST_EQ(px->use_count(), 2);
            }

            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_destructor

namespace n_assignment {

    void copy_assignment()
    {
        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> p1;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
            p1 = p1;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

            HPX_TEST_EQ(p1, p1);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            hpx::intrusive_ptr<X> p2;

            p1 = p2;

            HPX_TEST_EQ(p1, p2);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            hpx::intrusive_ptr<X> p3(p1);

            p1 = p3;

            HPX_TEST_EQ(p1, p3);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            HPX_TEST_EQ(N::base::instances, 0);

            hpx::intrusive_ptr<X> p4(new X);

            HPX_TEST_EQ(N::base::instances, 1);

            p1 = p4;

            HPX_TEST_EQ(N::base::instances, 1);

            HPX_TEST_EQ(p1, p4);

            HPX_TEST_EQ(p1->use_count(), 2);

            p1 = p2;

            HPX_TEST_EQ(p1, p2);
            HPX_TEST_EQ(N::base::instances, 1);

            p4 = p3;

            HPX_TEST_EQ(p4, p3);
            HPX_TEST_EQ(N::base::instances, 0);
        }
    }

    void conversion_assignment()
    {
        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> p1;

            hpx::intrusive_ptr<Y> p2;

            p1 = p2;

            HPX_TEST_EQ(p1, p2);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            HPX_TEST_EQ(N::base::instances, 0);

            hpx::intrusive_ptr<Y> p4(new Y);

            HPX_TEST_EQ(N::base::instances, 1);
            HPX_TEST_EQ(p4->use_count(), 1);

            hpx::intrusive_ptr<X> p5(p4);
            HPX_TEST_EQ(p4->use_count(), 2);

            p1 = p4;

            HPX_TEST_EQ(N::base::instances, 1);

            HPX_TEST_EQ(p1, p4);

            HPX_TEST_EQ(p1->use_count(), 3);
            HPX_TEST_EQ(p4->use_count(), 3);

            p1 = p2;

            HPX_TEST_EQ(p1, p2);
            HPX_TEST_EQ(N::base::instances, 1);
            HPX_TEST_EQ(p4->use_count(), 2);

            p4 = p2;
            p5 = p2;

            HPX_TEST_EQ(p4, p2);
            HPX_TEST_EQ(N::base::instances, 0);
        }
    }

    void pointer_assignment()
    {
        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> p1;

            p1 = p1.get();

            HPX_TEST_EQ(p1, p1);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            hpx::intrusive_ptr<X> p2;

            p1 = p2.get();

            HPX_TEST_EQ(p1, p2);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            hpx::intrusive_ptr<X> p3(p1);

            p1 = p3.get();

            HPX_TEST_EQ(p1, p3);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            HPX_TEST_EQ(N::base::instances, 0);

            hpx::intrusive_ptr<X> p4(new X);

            HPX_TEST_EQ(N::base::instances, 1);

            p1 = p4.get();

            HPX_TEST_EQ(N::base::instances, 1);

            HPX_TEST_EQ(p1, p4);

            HPX_TEST_EQ(p1->use_count(), 2);

            p1 = p2.get();

            HPX_TEST_EQ(p1, p2);
            HPX_TEST_EQ(N::base::instances, 1);

            p4 = p3.get();

            HPX_TEST_EQ(p4, p3);
            HPX_TEST_EQ(N::base::instances, 0);
        }

        {
            hpx::intrusive_ptr<X> p1;

            hpx::intrusive_ptr<Y> p2;

            p1 = p2.get();

            HPX_TEST_EQ(p1, p2);
            HPX_TEST(p1 ? false : true);
            HPX_TEST(!p1);
            HPX_TEST(p1.get() == nullptr);

            HPX_TEST_EQ(N::base::instances, 0);

            hpx::intrusive_ptr<Y> p4(new Y);

            HPX_TEST_EQ(N::base::instances, 1);
            HPX_TEST_EQ(p4->use_count(), 1);

            hpx::intrusive_ptr<X> p5(p4);
            HPX_TEST_EQ(p4->use_count(), 2);

            p1 = p4.get();

            HPX_TEST_EQ(N::base::instances, 1);

            HPX_TEST_EQ(p1, p4);

            HPX_TEST_EQ(p1->use_count(), 3);
            HPX_TEST_EQ(p4->use_count(), 3);

            p1 = p2.get();

            HPX_TEST_EQ(p1, p2);
            HPX_TEST_EQ(N::base::instances, 1);
            HPX_TEST_EQ(p4->use_count(), 2);

            p4 = p2.get();
            p5 = p2.get();

            HPX_TEST_EQ(p4, p2);
            HPX_TEST_EQ(N::base::instances, 0);
        }
    }

    void test()
    {
        copy_assignment();
        conversion_assignment();
        pointer_assignment();
    }

}    // namespace n_assignment

namespace n_reset {

    void test()
    {
        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px;
            HPX_TEST(px.get() == nullptr);

            px.reset();
            HPX_TEST(px.get() == nullptr);

            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);
            HPX_TEST_EQ(N::base::instances, 1);

            px.reset(p);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);

            px.reset();
            HPX_TEST(px.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr);
            HPX_TEST(px.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr, false);
            HPX_TEST(px.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST_EQ(N::base::instances, 1);

            px.reset(nullptr, true);
            HPX_TEST(px.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 1);

            hpx::intrusive_ptr<X> px;
            HPX_TEST(px.get() == nullptr);

            px.reset(p, true);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 1);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using hpx::intrusive_ptr_add_ref;
#endif
            intrusive_ptr_add_ref(p);
            HPX_TEST_EQ(p->use_count(), 1);

            hpx::intrusive_ptr<X> px;
            HPX_TEST(px.get() == nullptr);

            px.reset(p, false);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST(px.get() != nullptr);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 2);

            px.reset(p);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST(px.get() != nullptr);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            HPX_TEST_EQ(N::base::instances, 2);

            px.reset(p, true);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST(px.get() != nullptr);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);

            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using hpx::intrusive_ptr_add_ref;
#endif
            intrusive_ptr_add_ref(p);
            HPX_TEST_EQ(p->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 2);

            px.reset(p, false);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);

            HPX_TEST_EQ(N::base::instances, 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_reset

namespace n_access {

    void test()
    {
        {
            hpx::intrusive_ptr<X> px;
            HPX_TEST(px ? false : true);
            HPX_TEST(!px);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            HPX_TEST_EQ(get_pointer(px), px.get());
        }

        {
            hpx::intrusive_ptr<X> px(nullptr);
            HPX_TEST(px ? false : true);
            HPX_TEST(!px);

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            HPX_TEST_EQ(get_pointer(px), px.get());
        }

        {
            hpx::intrusive_ptr<X> px(new X);
            HPX_TEST(px ? true : false);
            HPX_TEST(!!px);
            HPX_TEST_EQ(&*px, px.get());
            HPX_TEST_EQ(px.operator->(), px.get());

#if defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
            using boost::get_pointer;
#endif

            HPX_TEST_EQ(get_pointer(px), px.get());
        }

        {
            hpx::intrusive_ptr<X> px;
            X* detached = px.detach();
            HPX_TEST(px.get() == nullptr);
            HPX_TEST(detached == nullptr);
        }

        {
            X* p = new X;
            HPX_TEST_EQ(p->use_count(), 0);

            hpx::intrusive_ptr<X> px(p);
            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 1);

            X* detached = px.detach();
            HPX_TEST(px.get() == nullptr);

            HPX_TEST_EQ(detached, p);
            HPX_TEST_EQ(detached->use_count(), 1);

            delete detached;
        }
    }

}    // namespace n_access

namespace n_swap {

    void test()
    {
        {
            hpx::intrusive_ptr<X> px;
            hpx::intrusive_ptr<X> px2;

            px.swap(px2);

            HPX_TEST(px.get() == nullptr);
            HPX_TEST(px2.get() == nullptr);

            using std::swap;
            swap(px, px2);

            HPX_TEST(px.get() == nullptr);
            HPX_TEST(px2.get() == nullptr);
        }

        {
            X* p = new X;
            hpx::intrusive_ptr<X> px;
            hpx::intrusive_ptr<X> px2(p);
            hpx::intrusive_ptr<X> px3(px2);

            px.swap(px2);

            HPX_TEST_EQ(px.get(), p);
            HPX_TEST_EQ(px->use_count(), 2);
            HPX_TEST(px2.get() == nullptr);
            HPX_TEST_EQ(px3.get(), p);
            HPX_TEST_EQ(px3->use_count(), 2);

            using std::swap;
            swap(px, px2);

            HPX_TEST(px.get() == nullptr);
            HPX_TEST_EQ(px2.get(), p);
            HPX_TEST_EQ(px2->use_count(), 2);
            HPX_TEST_EQ(px3.get(), p);
            HPX_TEST_EQ(px3->use_count(), 2);
        }

        {
            X* p1 = new X;
            X* p2 = new X;
            hpx::intrusive_ptr<X> px(p1);
            hpx::intrusive_ptr<X> px2(p2);
            hpx::intrusive_ptr<X> px3(px2);

            px.swap(px2);

            HPX_TEST_EQ(px.get(), p2);
            HPX_TEST_EQ(px->use_count(), 2);
            HPX_TEST_EQ(px2.get(), p1);
            HPX_TEST_EQ(px2->use_count(), 1);
            HPX_TEST_EQ(px3.get(), p2);
            HPX_TEST_EQ(px3->use_count(), 2);

            using std::swap;
            swap(px, px2);

            HPX_TEST_EQ(px.get(), p1);
            HPX_TEST_EQ(px->use_count(), 1);
            HPX_TEST_EQ(px2.get(), p2);
            HPX_TEST_EQ(px2->use_count(), 2);
            HPX_TEST_EQ(px3.get(), p2);
            HPX_TEST_EQ(px3->use_count(), 2);
        }
    }

}    // namespace n_swap

namespace n_comparison {

    template <class T, class U>
    void test2(hpx::intrusive_ptr<T> const& p, hpx::intrusive_ptr<U> const& q)
    {
        HPX_TEST((p == q) == (p.get() == q.get()));
        HPX_TEST((p != q) == (p.get() != q.get()));
    }

    template <class T>
    void test3(hpx::intrusive_ptr<T> const& p, hpx::intrusive_ptr<T> const& q)
    {
        HPX_TEST((p == q) == (p.get() == q.get()));
        HPX_TEST((p.get() == q) == (p.get() == q.get()));
        HPX_TEST((p == q.get()) == (p.get() == q.get()));
        HPX_TEST((p != q) == (p.get() != q.get()));
        HPX_TEST((p.get() != q) == (p.get() != q.get()));
        HPX_TEST((p != q.get()) == (p.get() != q.get()));

        // 'less' moved here as a g++ 2.9x parse error workaround
        std::less<T*> less;
        HPX_TEST((p < q) == less(p.get(), q.get()));
    }

    void test()
    {
        {
            hpx::intrusive_ptr<X> px;
            test3(px, px);

            hpx::intrusive_ptr<X> px2;
            test3(px, px2);

            hpx::intrusive_ptr<X> px3(px);
            test3(px3, px3);
            test3(px, px3);
        }

        {
            hpx::intrusive_ptr<X> px;

            hpx::intrusive_ptr<X> px2(new X);
            test3(px, px2);
            test3(px2, px2);

            hpx::intrusive_ptr<X> px3(new X);
            test3(px2, px3);

            hpx::intrusive_ptr<X> px4(px2);
            test3(px2, px4);
            test3(px4, px4);
        }

        {
            hpx::intrusive_ptr<X> px(new X);

            hpx::intrusive_ptr<Y> py(new Y);
            test2(px, py);

            hpx::intrusive_ptr<X> px2(py);
            test2(px2, py);
            test3(px, px2);
            test3(px2, px2);
        }
    }

}    // namespace n_comparison

namespace n_static_cast {

    void test()
    {
        {
            hpx::intrusive_ptr<X> px(new Y);

            hpx::intrusive_ptr<Y> py = hpx::static_pointer_cast<Y>(px);
            HPX_TEST_EQ(px.get(), py.get());
            HPX_TEST_EQ(px->use_count(), 2);
            HPX_TEST_EQ(py->use_count(), 2);

            hpx::intrusive_ptr<X> px2(py);
            HPX_TEST_EQ(px2.get(), px.get());
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<Y> py =
                hpx::static_pointer_cast<Y>(hpx::intrusive_ptr<X>(new Y));
            HPX_TEST(py.get() != nullptr);
            HPX_TEST_EQ(py->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_static_cast

namespace n_const_cast {

    void test()
    {
        {
            hpx::intrusive_ptr<X const> px;

            hpx::intrusive_ptr<X> px2 = hpx::const_pointer_cast<X>(px);
            HPX_TEST(px2.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<X> px2 =
                hpx::const_pointer_cast<X>(hpx::intrusive_ptr<X const>());
            HPX_TEST(px2.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X const> px(new X);

            hpx::intrusive_ptr<X> px2 = hpx::const_pointer_cast<X>(px);
            HPX_TEST_EQ(px2.get(), px.get());
            HPX_TEST_EQ(px2->use_count(), 2);
            HPX_TEST_EQ(px->use_count(), 2);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px =
                hpx::const_pointer_cast<X>(hpx::intrusive_ptr<X const>(new X));
            HPX_TEST(px.get() != nullptr);
            HPX_TEST_EQ(px->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_const_cast

namespace n_dynamic_cast {

    void test()
    {
        {
            hpx::intrusive_ptr<X> px;

            hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(px);
            HPX_TEST(py.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<Y> py =
                hpx::dynamic_pointer_cast<Y>(hpx::intrusive_ptr<X>());
            HPX_TEST(py.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<X> px(static_cast<X*>(nullptr));

            hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(px);
            HPX_TEST(py.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(
                hpx::intrusive_ptr<X>(static_cast<X*>(nullptr)));
            HPX_TEST(py.get() == nullptr);
        }

        {
            hpx::intrusive_ptr<X> px(new X);

            hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(px);
            HPX_TEST(py.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<Y> py =
                hpx::dynamic_pointer_cast<Y>(hpx::intrusive_ptr<X>(new X));
            HPX_TEST(py.get() == nullptr);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new Y);

            hpx::intrusive_ptr<Y> py = hpx::dynamic_pointer_cast<Y>(px);
            HPX_TEST_EQ(py.get(), px.get());
            HPX_TEST_EQ(py->use_count(), 2);
            HPX_TEST_EQ(px->use_count(), 2);
        }

        HPX_TEST_EQ(N::base::instances, 0);

        {
            hpx::intrusive_ptr<X> px(new Y);

            hpx::intrusive_ptr<Y> py =
                hpx::dynamic_pointer_cast<Y>(hpx::intrusive_ptr<X>(new Y));
            HPX_TEST(py.get() != nullptr);
            HPX_TEST_EQ(py->use_count(), 1);
        }

        HPX_TEST_EQ(N::base::instances, 0);
    }

}    // namespace n_dynamic_cast

namespace n_transitive {

    struct X : public N::base
    {
        hpx::intrusive_ptr<X> next;
    };

    void test()
    {
        hpx::intrusive_ptr<X> p(new X);
        p->next = hpx::intrusive_ptr<X>(new X);
        HPX_TEST(!p->next->next);
        p = p->next;
        HPX_TEST(!p->next);
    }

}    // namespace n_transitive

namespace n_report_1 {

    class foo : public N::base
    {
    public:
        foo()
          : m_self(this)
        {
        }

        void suicide()
        {
            m_self = nullptr;
        }

    private:
        hpx::intrusive_ptr<foo> m_self;
    };

    void test()
    {
        foo* foo_ptr = new foo;
        foo_ptr->suicide();
    }

}    // namespace n_report_1

int main()
{
    n_element_type::test();
    n_constructors::test();
    n_destructor::test();
    n_assignment::test();
    n_reset::test();
    n_access::test();
    n_swap::test();
    n_comparison::test();
    n_static_cast::test();
    n_const_cast::test();
    n_dynamic_cast::test();

    n_transitive::test();
    n_report_1::test();

    return hpx::util::report_errors();
}
