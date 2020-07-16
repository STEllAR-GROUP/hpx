//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_AWAIT) || defined(HPX_HAVE_CXX20_COROUTINES)

#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/memory.hpp>

#if defined(HPX_HAVE_CXX20_COROUTINES)
#include <coroutine>
#elif defined(HPX_HAVE_EMULATE_COROUTINE_SUPPORT_LIBRARY)
#include <hpx/util/await_traits.hpp>
#else
#include <experimental/coroutine>
#endif

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail {

#if defined(HPX_HAVE_CXX20_COROUTINES)
    template <typename Promise = void>
    using coroutine_handle = std::coroutine_handle<Promise>;
    using suspend_never = std::suspend_never;
#else
    template <typename Promise = void>
    using coroutine_handle = std::experimental::coroutine_handle<Promise>;
    using suspend_never = std::experimental::suspend_never;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // this was removed from the TS, so we define our own
    struct suspend_if
    {
        bool is_ready_;

        constexpr explicit suspend_if(bool cond) noexcept
          : is_ready_(!cond)
        {
        }

        constexpr bool await_ready() const noexcept
        {
            return is_ready_;
        }
        void await_suspend(coroutine_handle<>) const noexcept {}
        constexpr void await_resume() const noexcept {}
    };

    ///////////////////////////////////////////////////////////////////////////
    // Allow using co_await with an expression which evaluates to
    // hpx::future<T>.
    template <typename T>
    HPX_FORCEINLINE bool await_ready(future<T> const& f) noexcept
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    HPX_FORCEINLINE void await_suspend(
        future<T>& f, coroutine_handle<Promise> rh)
    {
        // f.then([=](future<T> result) {});
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed([st, rh]() mutable {
            if (st->has_exception())
            {
                rh.promise().set_exception(st->get_exception_ptr());
            }
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
    HPX_FORCEINLINE T await_resume(future<future<T>>& f)
    {
        return f.get().get();
    }

    template <typename T>
    HPX_FORCEINLINE T await_resume(future<shared_future<T>>& f)
    {
        return f.get().get();
    }

    // Allow using co_await with an expression which evaluates to
    // hpx::shared_future<T>.
    template <typename T>
    HPX_FORCEINLINE bool await_ready(shared_future<T> const& f) noexcept
    {
        return f.is_ready();
    }

    template <typename T, typename Promise>
    HPX_FORCEINLINE void await_suspend(
        shared_future<T>& f, coroutine_handle<Promise> rh)
    {
        // f.then([=](shared_future<T> result) {})
        auto st = traits::detail::get_shared_state(f);
        st->set_on_completed([st, rh]() mutable {
            if (st->has_exception())
            {
                rh.promise().set_exception(st->get_exception_ptr());
            }
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
    template <typename Allocator, typename T, typename Derived>
    struct coroutine_promise_base
      : hpx::lcos::detail::future_data_allocator<T, Allocator, Derived>
    {
        using base_type =
            hpx::lcos::detail::future_data_allocator<T, Allocator, Derived>;
        using init_no_addref = typename base_type::init_no_addref;

        typedef typename std::allocator_traits<
            Allocator>::template rebind_alloc<Derived>
            other_allocator;

        // the shared state is held alive by the coroutine
        coroutine_promise_base()
          : base_type(init_no_addref{}, other_allocator{})
        {
        }

        coroutine_promise_base(other_allocator const& alloc)
          : base_type(init_no_addref{}, alloc)
        {
        }

        hpx::lcos::future<T> get_return_object()
        {
            hpx::intrusive_ptr<Derived> shared_state(
                static_cast<Derived*>(this));
            return hpx::traits::future_access<hpx::lcos::future<T>>::create(
                std::move(shared_state));
        }

        constexpr suspend_never initial_suspend() const noexcept
        {
            return suspend_never{};
        }

        suspend_if final_suspend() noexcept
        {
            // This gives up the coroutine's reference count on the shared
            // state. If this was the last reference count, the coroutine
            // should not suspend before exiting.
            return suspend_if{!this->base_type::requires_delete()};
        }

        void destroy() override
        {
            coroutine_handle<Derived>::from_promise(
                *static_cast<Derived*>(this))
                .destroy();
        }

        // allocator support for shared coroutine state
        HPX_NODISCARD static void* allocate(std::size_t size)
        {
            using char_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<char>;

            using traits = std::allocator_traits<char_allocator>;
            using unique_ptr =
                std::unique_ptr<char, util::allocator_deleter<char_allocator>>;

            char_allocator alloc{};
            unique_ptr p(traits::allocate(alloc, size),
                util::allocator_deleter<char_allocator>{alloc});

            using derived_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<Derived>;

            derived_allocator alloc_derived{};
            traits::construct(
                alloc, reinterpret_cast<Derived*>(p.get()), alloc_derived);

            return p.release();
        }

        HPX_FORCEINLINE void call_base_destroy()
        {
            this->base_type::destroy();
        }

        static void deallocate(void* p, std::size_t size) noexcept
        {
            // the destroy() of the base takes care of the memory
            static_cast<Derived*>(p)->call_base_destroy();
        }
    };
}}}    // namespace hpx::lcos::detail

///////////////////////////////////////////////////////////////////////////////
namespace std {
#if !defined(HPX_HAVE_CXX20_COROUTINES)
    namespace experimental {
#endif
        // Allow for functions which use co_await to return an hpx::future<T>
        template <typename T, typename... Ts>
        struct coroutine_traits<hpx::lcos::future<T>, Ts...>
        {
            using allocator_type =
                hpx::util::internal_allocator<coroutine_traits>;

            struct promise_type
              : hpx::lcos::detail::coroutine_promise_base<allocator_type, T,
                    promise_type>
            {
                using base_type =
                    hpx::lcos::detail::coroutine_promise_base<allocator_type, T,
                        promise_type>;

                promise_type() = default;

                template <typename Allocator>
                promise_type(Allocator const& alloc)
                  : base_type(alloc)
                {
                }

                template <typename U>
                void return_value(U&& value)
                {
                    this->base_type::set_value(std::forward<U>(value));
                }

                HPX_NODISCARD HPX_FORCEINLINE static void* operator new(
                    std::size_t size)
                {
                    return base_type::allocate(size);
                }

                HPX_FORCEINLINE static void operator delete(
                    void* p, std::size_t size) noexcept
                {
                    base_type::deallocate(p, size);
                }
            };
        };

        template <typename... Ts>
        struct coroutine_traits<hpx::lcos::future<void>, Ts...>
        {
            using allocator_type =
                hpx::util::internal_allocator<coroutine_traits>;

            struct promise_type
              : hpx::lcos::detail::coroutine_promise_base<allocator_type, void,
                    promise_type>
            {
                using base_type =
                    hpx::lcos::detail::coroutine_promise_base<allocator_type,
                        void, promise_type>;

                promise_type() = default;

                template <typename Allocator>
                promise_type(Allocator const& alloc)
                  : base_type(alloc)
                {
                }

                void return_void()
                {
                    this->base_type::set_value();
                }

                HPX_NODISCARD HPX_FORCEINLINE static void* operator new(
                    std::size_t size)
                {
                    return base_type::allocate();
                }

                HPX_FORCEINLINE static void operator delete(
                    void* p, std::size_t size) noexcept
                {
                    base_type::deallocate(p, size);
                }
            };
        };

        // Allow for functions which use co_await to return an
        // hpx::shared_future<T>
        template <typename T, typename... Ts>
        struct coroutine_traits<hpx::lcos::shared_future<T>, Ts...>
        {
            using allocator_type =
                hpx::util::internal_allocator<coroutine_traits>;

            struct promise_type
              : hpx::lcos::detail::coroutine_promise_base<allocator_type, T,
                    promise_type>
            {
                using base_type =
                    hpx::lcos::detail::coroutine_promise_base<allocator_type, T,
                        promise_type>;

                promise_type() = default;

                template <typename Allocator>
                promise_type(Allocator const& alloc)
                  : base_type(alloc)
                {
                }

                template <typename U>
                void return_value(U&& value)
                {
                    this->base_type::set_value(std::forward<U>(value));
                }

                HPX_NODISCARD HPX_FORCEINLINE static void* operator new(
                    std::size_t size)
                {
                    return base_type::allocate(size);
                }

                HPX_FORCEINLINE static void operator delete(
                    void* p, std::size_t size) noexcept
                {
                    base_type::deallocate(p, size);
                }
            };
        };

        template <typename... Ts>
        struct coroutine_traits<hpx::lcos::shared_future<void>, Ts...>
        {
            using allocator_type =
                hpx::util::internal_allocator<coroutine_traits>;

            struct promise_type
              : hpx::lcos::detail::coroutine_promise_base<allocator_type, void,
                    promise_type>
            {
                using base_type =
                    hpx::lcos::detail::coroutine_promise_base<allocator_type,
                        void, promise_type>;

                promise_type() = default;

                template <typename Allocator>
                promise_type(Allocator const& alloc)
                  : base_type(alloc)
                {
                }

                void return_void()
                {
                    this->base_type::set_value();
                }

                HPX_NODISCARD HPX_FORCEINLINE static void* operator new(
                    std::size_t size)
                {
                    return base_type::allocate();
                }

                HPX_FORCEINLINE static void operator delete(
                    void* p, std::size_t size) noexcept
                {
                    base_type::deallocate(p, size);
                }
            };
        };
#if !defined(HPX_HAVE_CXX20_COROUTINES)
    }    // namespace experimental
#endif
}    // namespace std

#endif    // HPX_HAVE_AWAIT || HPX_HAVE_CXX20_COROUTINES
