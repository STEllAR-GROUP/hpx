//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C)
#define HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C

#if defined(HPX_HAVE_AWAIT)

#include <hpx/config.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/traits/future_access.hpp>

#if defined(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)
#include <hpx/util/await_traits.hpp>
#else
#include <experimental/coroutine>
#endif

#include <boost/intrusive_ptr.hpp>

#include <exception>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // this was removed from the TS, so we define our own
    struct suspend_if
    {
        bool is_ready_;

        explicit suspend_if(bool cond) noexcept : is_ready_(!cond) {}

        bool await_ready() noexcept { return is_ready_; }
        void await_suspend(std::experimental::coroutine_handle<>) noexcept {}
        void await_resume() noexcept {}
    };

    ///////////////////////////////////////////////////////////////////////////
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
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed(
            [=]() mutable
            {
                if (st->has_exception())
                    rh.promise().set_exception(st->get_exception_ptr());
                rh();
            });
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
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed(
            [=]() mutable
            {
                if (st->has_exception())
                    rh.promise().set_exception(st->get_exception_ptr());
                rh();
            });
    }

    template <typename T>
    HPX_FORCEINLINE T await_resume(shared_future<T>& f)
    {
        return f.get();
    }

    ///////////////////////////////////////////////////////////////////////////
    // derive from future shared state as this will be combined with the
    // necessary stack frame for the resumable function
    template <typename T, typename Derived>
    struct coroutine_promise_base : hpx::lcos::detail::future_data<T>
    {
        typedef hpx::lcos::detail::future_data<T> base_type;

        coroutine_promise_base()
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

        std::experimental::suspend_never initial_suspend()
        {
            return std::experimental::suspend_never{};
        }

        suspend_if final_suspend()
        {
            // This gives up the coroutine's reference count on the shared
            // state. If this was the last reference count, the coroutine
            // should not suspend before exiting.
            return suspend_if{!this->base_type::requires_delete()};
        }

        void set_exception(std::exception_ptr e)
        {
            try {
                std::rethrow_exception(std::move(e));
            }
            catch (...) {
                this->base_type::set_exception(std::current_exception());
            }
        }

        void destroy()
        {
            std::experimental::coroutine_handle<Derived>::
                from_promise(*static_cast<Derived*>(this)).destroy();
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace std { namespace experimental
{
    // Allow for functions which use co_await to return an hpx::future<T>
    template <typename T, typename ...Ts>
    struct coroutine_traits<hpx::lcos::future<T>, Ts...>
    {
        struct promise_type
          : hpx::lcos::detail::coroutine_promise_base<T, promise_type>
        {
            using base_type =
                hpx::lcos::detail::coroutine_promise_base<T, promise_type>;

            template <typename U>
            void return_value(U && value)
            {
                this->base_type::set_value(std::forward<U>(value));
            }
        };
    };

    template <typename ...Ts>
    struct coroutine_traits<hpx::lcos::future<void>, Ts...>
    {
        struct promise_type
          : hpx::lcos::detail::coroutine_promise_base<void, promise_type>
        {
            using base_type =
                hpx::lcos::detail::coroutine_promise_base<void, promise_type>;

            void return_void()
            {
                this->base_type::set_value();
            }
        };
    };
}}

#endif // HPX_HAVE_AWAIT
#endif // HPX_LCOS_ACBA3E3F_7B29_41D1_AE85_C73CB69D089C

