//  Copyright (c) 2016 Gor Nishanov
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Partial implementation of the coroutine support library
// (<experimental/coroutines>).

#if !defined(HPX_UTIL_933DADE6_D0AB_46EB_98F3_9ECDAE31A4DA)
#define HPX_UTIL_933DADE6_D0AB_46EB_98F3_9ECDAE31A4DA

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)

namespace std { namespace experimental
{
    template <typename R, typename...>
    struct coroutine_traits
    {
        using promise_type = typename R::promise_type;
    };

    template <typename Promise = void>
    struct coroutine_handle;

    template <>
    struct coroutine_handle<void>
    {
        static coroutine_handle from_address(void *addr) noexcept
        {
            coroutine_handle me;
            me.ptr = addr;
            return me;
        }
        void operator()()
        {
            resume();
        }
        void *address() const
        {
            return ptr;
        }
        void resume() const
        {
            __builtin_coro_resume(ptr);
        }
        void destroy() const
        {
            __builtin_coro_destroy(ptr);
        }
        bool done() const
        {
            return __builtin_coro_done(ptr);
        }
        coroutine_handle &operator=(decltype(nullptr))
        {
            ptr = nullptr;
            return *this;
        }
        coroutine_handle(decltype(nullptr))
          : ptr(nullptr)
        {
        }
        coroutine_handle()
          : ptr(nullptr)
        {
        }
        //  void reset() { ptr = nullptr; } // add to P0057?
        explicit operator bool() const
        {
            return ptr;
        }

    protected:
        void *ptr;
    };

    template <typename Promise>
    struct coroutine_handle : coroutine_handle<>
    {
        using coroutine_handle<>::operator=;

        static coroutine_handle from_address(void *addr) noexcept
        {
            coroutine_handle me;
            me.ptr = addr;
            return me;
        }

        Promise &promise() const
        {
            return *reinterpret_cast<Promise *>(
                __builtin_coro_promise(ptr, alignof(Promise), false));
        }
        static coroutine_handle from_promise(Promise &promise)
        {
            coroutine_handle p;
            p.ptr = __builtin_coro_promise(&promise, alignof(Promise), true);
            return p;
        }
    };

    template <typename Promise>
    bool operator==(coroutine_handle<Promise> const &left,
        coroutine_handle<Promise> const &right) noexcept
    {
        return left.address() == right.address();
    }

    template <typename Promise>
    bool operator!=(coroutine_handle<Promise> const &left,
        coroutine_handle<Promise> const &right) noexcept
    {
        return !(left == right);
    }

    struct suspend_always
    {
        bool await_ready() const
        {
            return false;
        }
        void await_suspend(coroutine_handle<>) const
        {
        }
        void await_resume() const
        {
        }
    };

    struct suspend_never
    {
        bool await_ready() const
        {
            return true;
        }
        void await_suspend(coroutine_handle<>) const
        {
        }
        void await_resume() const
        {
        }
    };
}}

#endif    // defined(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)
#endif    // HPX_UTIL_933DADE6_D0AB_46EB_98F3_9ECDAE31A4DA
