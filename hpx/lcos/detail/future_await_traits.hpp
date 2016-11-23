//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C)
#define HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C

#if defined(HPX_HAVE_AWAIT)

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/traits/future_access.hpp>

#include <boost/intrusive_ptr.hpp>

#if defined(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)
#include <hpx/util/await_traits.hpp>
#else
#include <experimental/coroutine>
#endif

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    // Allow using co_await with an expression which evaluates to
    // hpx::future<T>.
    template <typename T>
    HPX_FORCEINLINE bool await_ready(future<T> const& f)
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    HPX_FORCEINLINE void await_suspend(future<T>& f,
        std::experimental::coroutine_handle<Promise> rh)
    {
        // f.then([=](future<T> result) {});
        traits::detail::get_shared_state(f)->set_on_completed(rh);
    }

    template <typename T>
    HPX_FORCEINLINE T await_resume(future<T>& f)
    {
        return f.get();
    }

    // Allow wrapped futures to be unwrapped, if possible.
    template <typename T>
    HPX_FORCEINLINE T await_resume(future<future<T> >& f)
    {
        return f.get().get();
    }

    template <typename T>
    HPX_FORCEINLINE T await_resume(future<shared_future<T> >& f)
    {
        return f.get().get();
    }

    // Allow using co_await with an expression which evaluates to
    // hpx::shared_future<T>.
    template <typename T>
    HPX_FORCEINLINE bool await_ready(shared_future<T> const& f)
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    HPX_FORCEINLINE void await_suspend(shared_future<T>& f,
        std::experimental::coroutine_handle<Promise> rh)
    {
        // f.then([=](shared_future<T> result) {})
        traits::detail::get_shared_state(f)->set_on_completed(rh);
    }

    template <typename T>
    HPX_FORCEINLINE T await_resume(shared_future<T>& f)
    {
        return f.get();
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace std { namespace experimental
{
    // Allow for functions which use co_await to return an hpx::future<T>
    template <typename T, typename ...Ts>
    struct coroutine_traits<hpx::lcos::future<T>, Ts...>
    {
        // derive from future shared state as this will be combined with the
        // necessary stack frame for the resumable function
        struct promise_type : hpx::lcos::detail::future_data<T>
        {
            typedef hpx::lcos::detail::future_data<T> base_type;

            promise_type()
            {
                // the shared state is held alive by the coroutine
                hpx::lcos::detail::intrusive_ptr_add_ref(this);
            }

            hpx::lcos::future<T> get_return_object()
            {
                boost::intrusive_ptr<base_type> shared_state(this);
                return hpx::traits::future_access<hpx::lcos::future<T> >::
                    create(std::move(shared_state));
            }

            std::experimental::suspend_never initial_suspend() { return {}; }

            std::experimental::suspend_if final_suspend()
            {
                // This gives up the coroutine's reference count on the shared
                // state. If this was the last reference count, the coroutine
                // should not suspend before exiting.
                return {!this->base_type::requires_delete()};
            }

            template <typename U, typename U2 = T, typename V =
                typename std::enable_if<!std::is_void<U2>::value>::type>
            void return_value(U && value)
            {
                this->base_type::set_value(std::forward<U>(value));
            }

            template <typename U = T, typename V =
                typename std::enable_if<std::is_void<U>::value>::type>
            void return_value()
            {
                this->base_type::set_value();
            }

            void set_exception(std::exception_ptr e)
            {
                try {
                    std::rethrow_exception(e);
                }
                catch (...) {
                    this->base_type::set_exception(boost::current_exception());
                }
            }

            void destroy()
            {
                coroutine_handle<promise_type>::from_promise(*this).destroy();
            }
        };
    };
}}

#endif // HPX_HAVE_AWAIT
#endif // HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C

