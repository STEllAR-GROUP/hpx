//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/polymorphic_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/thread_support.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution {

    namespace detail {

        struct shape_iter_impl_base;
        HPX_PARALLELISM_EXPORT void intrusive_ptr_add_ref(
            shape_iter_impl_base* p);
        HPX_PARALLELISM_EXPORT void intrusive_ptr_release(
            shape_iter_impl_base* p);

        struct shape_iter_impl_base
        {
            virtual ~shape_iter_impl_base() = default;
            virtual void copy(shape_iter_impl_base const&) = 0;
            virtual shape_iter_impl_base* clone() const = 0;
            virtual void increment() = 0;
            virtual std::size_t dereference() const = 0;
            virtual bool equal_to(shape_iter_impl_base const&) const = 0;
        };

        template <typename Iterator>
        struct shape_iter_impl : shape_iter_impl_base
        {
            shape_iter_impl(Iterator it)
              : iterator_(it)
            {
            }

            void copy(shape_iter_impl_base const& from) override
            {
                iterator_ = static_cast<shape_iter_impl const&>(from).iterator_;
            }

            shape_iter_impl_base* clone() const override
            {
                return new shape_iter_impl(iterator_);
            }

            void increment() override
            {
                ++iterator_;
            }

            std::size_t dereference() const override
            {
                return *iterator_;
            }

            bool equal_to(shape_iter_impl_base const& other) const override
            {
                return iterator_ ==
                    static_cast<shape_iter_impl const&>(other).iterator_;
            }

        private:
            Iterator iterator_;
        };

        struct shape_iter
        {
            using iterator_category = std::input_iterator_tag;
            using value_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using pointer = std::size_t*;
            using reference = std::size_t&;

            template <typename Iterator>
            shape_iter(Iterator it)
              : impl_(new shape_iter_impl<Iterator>(it))
            {
            }

            // we need to deep-copy the embedded iterator
            shape_iter(shape_iter const& rhs)
              : impl_(rhs.impl_->clone())
            {
            }

            shape_iter& operator=(shape_iter const& rhs)
            {
                impl_->copy(*rhs.impl_);
                return *this;
            }

            std::size_t operator*() const
            {
                return impl_->dereference();
            }

            shape_iter& operator++()
            {
                impl_->increment();
                return *this;
            }

            shape_iter operator++(int)
            {
                auto copy = *this;
                ++*this;
                return copy;
            }

            friend bool operator==(shape_iter const& lhs, shape_iter const& rhs)
            {
                return lhs.impl_->equal_to(*rhs.impl_);
            }

            friend bool operator!=(shape_iter const& lhs, shape_iter const& rhs)
            {
                return !(lhs == rhs);
            }

        protected:
            std::unique_ptr<shape_iter_impl_base> impl_;
        };

        struct range_proxy
        {
            using value_type = std::size_t;

            template <typename Shape>
            range_proxy(Shape const& s)
              : begin_(s.begin())
              , end_(s.end())
              , size_(std::distance(s.begin(), s.end()))
            {
            }

            shape_iter begin() const
            {
                return begin_;
            }

            shape_iter end() const
            {
                return end_;
            }

            std::ptrdiff_t size() const noexcept
            {
                return size_;
            }

        private:
            shape_iter begin_;
            shape_iter end_;
            std::ptrdiff_t size_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct construct_vtable
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename VTable, typename T>
        struct polymorphic_executor_vtables
        {
            static constexpr VTable instance = detail::construct_vtable<T>();
        };

        template <typename VTable, typename T>
        constexpr VTable polymorphic_executor_vtables<VTable, T>::instance;

        template <typename VTable, typename T>
        constexpr VTable const* get_polymorphic_executor_vtable() noexcept
        {
            static_assert(
                !std::is_reference<T>::value, "T shall have no ref-qualifiers");

            return &polymorphic_executor_vtables<VTable, T>::instance;
        }

        ///////////////////////////////////////////////////////////////////////
        struct empty_polymorphic_executor
        {
        };    // must be trivial and empty

        HPX_NORETURN HPX_PARALLELISM_EXPORT void
        throw_bad_polymorphic_executor();

        template <typename R>
        HPX_NORETURN inline R throw_bad_polymorphic_executor()
        {
            throw_bad_polymorphic_executor();
        }

        ///////////////////////////////////////////////////////////////////////
        struct vtable_base
        {
            template <typename T>
            static T& get(void* obj) noexcept
            {
                return *reinterpret_cast<T*>(obj);
            }

            template <typename T>
            static T const& get(void const* obj) noexcept
            {
                return *reinterpret_cast<T const*>(obj);
            }

            template <typename T>
            static void* allocate(void* storage, std::size_t storage_size)
            {
                using storage_t =
                    typename std::aligned_storage<sizeof(T), alignof(T)>::type;

                if (sizeof(T) > storage_size)
                {
                    return new storage_t;
                }
                return storage;
            }

            template <typename T>
            static void _deallocate(
                void* obj, std::size_t storage_size, bool destroy)
            {
                using storage_t =
                    typename std::aligned_storage<sizeof(T), alignof(T)>::type;

                if (destroy)
                {
                    get<T>(obj).~T();
                }

                if (sizeof(T) > storage_size)
                {
                    delete static_cast<storage_t*>(obj);
                }
            }
            void (*deallocate)(void*, std::size_t storage_size, bool);

            template <typename T>
            static void* _copy(void* storage, std::size_t storage_size,
                void const* src, bool destroy)
            {
                if (destroy)
                    get<T>(storage).~T();

                void* buffer = allocate<T>(storage, storage_size);
                return ::new (buffer) T(get<T>(src));
            }
            void* (*copy)(void*, std::size_t, void const*, bool);

            template <typename T>
            constexpr vtable_base(construct_vtable<T>) noexcept
              : deallocate(&vtable_base::template _deallocate<T>)
              , copy(&vtable_base::template _copy<T>)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // NonBlockingOneWayExecutor interface
        template <typename Sig>
        struct never_blocking_oneway_vtable;

        template <typename R, typename... Ts>
        struct never_blocking_oneway_vtable<R(Ts...)>
        {
            using post_function_type =
                hpx::util::unique_function_nonser<R(Ts...)>;

            // post
            template <typename T>
            static void _post(void* exec, post_function_type&& f, Ts&&... ts)
            {
                execution::post(vtable_base::get<T>(exec), std::move(f),
                    std::forward<Ts>(ts)...);
            }
            void (*post)(void*, post_function_type&&, Ts&&...);

            template <typename T>
            constexpr never_blocking_oneway_vtable(construct_vtable<T>) noexcept
              : post(&never_blocking_oneway_vtable::template _post<T>)
            {
            }

            static void _empty_post(void*, post_function_type&&, Ts&&...)
            {
                throw_bad_polymorphic_executor();
            }

            constexpr never_blocking_oneway_vtable(
                construct_vtable<empty_polymorphic_executor>) noexcept
              : post(&never_blocking_oneway_vtable::_empty_post)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // OneWayExecutor interface
        template <typename Sig>
        struct oneway_vtable;

        template <typename R, typename... Ts>
        struct oneway_vtable<R(Ts...)>
        {
            using sync_execute_function_type =
                hpx::util::unique_function_nonser<R(Ts...)>;

            // sync_execute
            template <typename T>
            static R _sync_execute(
                void* exec, sync_execute_function_type&& f, Ts&&... ts)
            {
                return execution::sync_execute(vtable_base::get<T>(exec),
                    std::move(f), std::forward<Ts>(ts)...);
            }
            R (*sync_execute)(void*, sync_execute_function_type&&, Ts&&...);

            template <typename T>
            constexpr oneway_vtable(construct_vtable<T>) noexcept
              : sync_execute(&oneway_vtable::template _sync_execute<T>)
            {
            }

            static R _empty_sync_execute(
                void*, sync_execute_function_type&&, Ts&&...)
            {
                throw_bad_polymorphic_executor<R>();
            }

            constexpr oneway_vtable(
                construct_vtable<empty_polymorphic_executor>) noexcept
              : sync_execute(&oneway_vtable::_empty_sync_execute)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // TwoWayExecutor interface
        template <typename Sig>
        struct twoway_vtable;

        template <typename R, typename... Ts>
        struct twoway_vtable<R(Ts...)>
        {
            using async_execute_function_type =
                hpx::util::unique_function_nonser<R(Ts...)>;
            using then_execute_function_type =
                hpx::util::unique_function_nonser<R(
                    hpx::shared_future<void> const&, Ts...)>;

            // async_execute
            template <typename T>
            static hpx::future<R> _async_execute(
                void* exec, async_execute_function_type&& f, Ts&&... ts)
            {
                return execution::async_execute(vtable_base::get<T>(exec),
                    std::move(f), std::forward<Ts>(ts)...);
            }
            hpx::future<R> (*async_execute)(
                void*, async_execute_function_type&& f, Ts&&...);

            // then_execute
            template <typename T>
            static hpx::future<R> _then_execute(void* exec,
                then_execute_function_type&& f,
                hpx::shared_future<void> const& predecessor, Ts&&... ts)
            {
                return execution::then_execute(vtable_base::get<T>(exec),
                    std::move(f), predecessor, std::forward<Ts>(ts)...);
            }
            hpx::future<R> (*then_execute)(void*,
                then_execute_function_type&& f, hpx::shared_future<void> const&,
                Ts&&...);

            template <typename T>
            constexpr twoway_vtable(construct_vtable<T>) noexcept
              : async_execute(&twoway_vtable::template _async_execute<T>)
              , then_execute(&twoway_vtable::template _then_execute<T>)
            {
            }

            static hpx::future<R> _empty_async_execute(void* /* exec */,
                async_execute_function_type&& /* f */, Ts&&... /* ts */)
            {
                throw_bad_polymorphic_executor<R>();
            }
            static hpx::future<R> _empty_then_execute(void*,
                then_execute_function_type&&, hpx::shared_future<void> const&,
                Ts&&...)
            {
                throw_bad_polymorphic_executor<R>();
            }

            constexpr twoway_vtable(
                construct_vtable<empty_polymorphic_executor>) noexcept
              : async_execute(&twoway_vtable::_empty_async_execute)
              , then_execute(&twoway_vtable::_empty_then_execute)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // BulkOneWayExecutor interface
        template <typename Sig>
        struct bulk_oneway_vtable;

        template <typename R, typename... Ts>
        struct bulk_oneway_vtable<R(Ts...)>
        {
            using bulk_sync_execute_function_type =
                hpx::util::function_nonser<R(std::size_t, Ts...)>;

            // bulk_sync_execute
            template <typename T>
            static std::vector<R> _bulk_sync_execute(void* exec,
                bulk_sync_execute_function_type&& f, range_proxy const& shape,
                Ts&&... ts)
            {
                return execution::bulk_sync_execute(vtable_base::get<T>(exec),
                    std::move(f), shape, std::forward<Ts>(ts)...);
            }
            std::vector<R> (*bulk_sync_execute)(void*,
                bulk_sync_execute_function_type&&, range_proxy const& shape,
                Ts&&...);

            template <typename T>
            constexpr bulk_oneway_vtable(construct_vtable<T>) noexcept
              : bulk_sync_execute(
                    &bulk_oneway_vtable::template _bulk_sync_execute<T>)
            {
            }

            static std::vector<R> _empty_bulk_sync_execute(void*,
                bulk_sync_execute_function_type&&, range_proxy const&, Ts&&...)
            {
                throw_bad_polymorphic_executor<R>();
            }

            constexpr bulk_oneway_vtable(
                construct_vtable<empty_polymorphic_executor>) noexcept
              : bulk_sync_execute(&bulk_oneway_vtable::_empty_bulk_sync_execute)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // BulkTwoWayExecutor interface
        template <typename Sig>
        struct bulk_twoway_vtable;

        template <typename R, typename... Ts>
        struct bulk_twoway_vtable<R(Ts...)>
        {
            using bulk_async_execute_function_type =
                hpx::util::function_nonser<R(std::size_t, Ts...)>;
            using bulk_then_execute_function_type =
                hpx::util::function_nonser<R(
                    std::size_t, hpx::shared_future<void> const&, Ts...)>;

            // bulk_async_execute
            template <typename T>
            static std::vector<hpx::future<R>> _bulk_async_execute(void* exec,
                bulk_async_execute_function_type&& f, range_proxy const& shape,
                Ts&&... ts)
            {
                return execution::bulk_async_execute(vtable_base::get<T>(exec),
                    std::move(f), shape, std::forward<Ts>(ts)...);
            }
            std::vector<hpx::future<R>> (*bulk_async_execute)(void*,
                bulk_async_execute_function_type&&, range_proxy const& shape,
                Ts&&...);

            // bulk_then_execute
            template <typename T>
            static hpx::future<std::vector<R>> _bulk_then_execute(void* exec,
                bulk_then_execute_function_type&& f, range_proxy const& shape,
                hpx::shared_future<void> const& predecessor, Ts&&... ts)
            {
                return execution::bulk_then_execute(vtable_base::get<T>(exec),
                    std::move(f), shape, predecessor, std::forward<Ts>(ts)...);
            }
            hpx::future<std::vector<R>> (*bulk_then_execute)(void*,
                bulk_then_execute_function_type&&, range_proxy const& shape,
                hpx::shared_future<void> const&, Ts&&...);

            template <typename T>
            constexpr bulk_twoway_vtable(construct_vtable<T>) noexcept
              : bulk_async_execute(
                    &bulk_twoway_vtable::template _bulk_async_execute<T>)
              , bulk_then_execute(
                    &bulk_twoway_vtable::template _bulk_then_execute<T>)
            {
            }

            static std::vector<hpx::future<R>> _empty_bulk_async_execute(void*,
                bulk_async_execute_function_type&&, range_proxy const&, Ts&&...)
            {
                throw_bad_polymorphic_executor<R>();
            }

            static hpx::future<std::vector<R>> _empty_bulk_then_execute(void*,
                bulk_then_execute_function_type&&, range_proxy const&,
                hpx::shared_future<void> const&, Ts&&...)
            {
                throw_bad_polymorphic_executor<R>();
            }

            constexpr bulk_twoway_vtable(
                construct_vtable<empty_polymorphic_executor>) noexcept
              : bulk_async_execute(
                    &bulk_twoway_vtable::_empty_bulk_async_execute)
              , bulk_then_execute(&bulk_twoway_vtable::_empty_bulk_then_execute)
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Sig>
        struct polymorphic_executor_vtable
          : vtable_base
          , never_blocking_oneway_vtable<Sig>
          , oneway_vtable<Sig>
          , twoway_vtable<Sig>
          , bulk_oneway_vtable<Sig>
          , bulk_twoway_vtable<Sig>
        {
            template <typename T>
            constexpr polymorphic_executor_vtable(construct_vtable<T>) noexcept
              : vtable_base(construct_vtable<T>())
              , never_blocking_oneway_vtable<Sig>(construct_vtable<T>())
              , oneway_vtable<Sig>(construct_vtable<T>())
              , twoway_vtable<Sig>(construct_vtable<T>())
              , bulk_oneway_vtable<Sig>(construct_vtable<T>())
              , bulk_twoway_vtable<Sig>(construct_vtable<T>())
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // make sure the empty table instance is initialized in time, even
        // during early startup
        template <typename Sig>
        struct polymorphic_executor_vtable;

        // NOTE: nvcc (at least CUDA 9.2 and 10.1) fails with an internal
        // compiler error ("there was an error in verifying the lgenfe output!")
        // with this enabled, so we explicitly use the fallback.
#if !defined(HPX_HAVE_GPU_SUPPORT)
        template <typename Sig>
        constexpr polymorphic_executor_vtable<Sig> const*
        get_empty_polymorphic_executor_vtable() noexcept
        {
            return &polymorphic_executor_vtables<
                polymorphic_executor_vtable<Sig>,
                empty_polymorphic_executor>::instance;
        }
#else
        template <typename Sig>
        polymorphic_executor_vtable<Sig> const*
        get_empty_polymorphic_executor_vtable() noexcept
        {
            static polymorphic_executor_vtable<Sig> const empty_vtable =
                detail::construct_vtable<empty_polymorphic_executor>();
            return &empty_vtable;
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        static constexpr std::size_t polymorphic_executor_storage_size =
            3 * sizeof(void*);

        class HPX_PARALLELISM_EXPORT polymorphic_executor_base
        {
            using vtable = vtable_base;

        public:
            constexpr explicit polymorphic_executor_base(
                vtable const* empty_vptr) noexcept
              : vptr(empty_vptr)
              , object(nullptr)
              , storage_init()
            {
            }

            polymorphic_executor_base(polymorphic_executor_base const& other,
                vtable const* empty_vtable);
            polymorphic_executor_base(polymorphic_executor_base&& other,
                vtable const* empty_vtable) noexcept;
            ~polymorphic_executor_base();

            void op_assign(polymorphic_executor_base const& other,
                vtable const* empty_vtable);
            void op_assign(polymorphic_executor_base&& other,
                vtable const* empty_vtable) noexcept;

            void destroy() noexcept;
            void reset(vtable const* empty_vptr) noexcept;
            void swap(polymorphic_executor_base& exec) noexcept;

            bool empty() const noexcept
            {
                return object == nullptr;
            }

            explicit operator bool() const noexcept
            {
                return !empty();
            }

        protected:
            vtable const* vptr;
            void* object;
            union
            {
                char storage_init;
                mutable unsigned char
                    storage[polymorphic_executor_storage_size];
            };
        };

        ///////////////////////////////////////////////////////////////////////////
        inline bool is_empty_polymorphic_executor_impl(
            polymorphic_executor_base const* exec) noexcept
        {
            return exec->empty();
        }

        inline constexpr bool is_empty_polymorphic_executor_impl(...) noexcept
        {
            return false;
        }

        template <typename Exec>
        constexpr bool is_empty_polymorphic_executor(Exec const& exec) noexcept
        {
            return detail::is_empty_polymorphic_executor_impl(&exec);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class polymorphic_executor;

    template <typename R, typename... Ts>
    class polymorphic_executor<R(Ts...)> : detail::polymorphic_executor_base
    {
        using base_type = detail::polymorphic_executor_base;
        using vtable = detail::polymorphic_executor_vtable<R(Ts...)>;

    public:
        constexpr polymorphic_executor() noexcept
          : base_type(get_empty_vtable())
        {
        }

        polymorphic_executor(polymorphic_executor const& other)
          : base_type(other, get_empty_vtable())
        {
        }

        polymorphic_executor(polymorphic_executor&& other) noexcept
          : base_type(std::move(other), get_empty_vtable())
        {
        }

        polymorphic_executor& operator=(polymorphic_executor const& other)
        {
            base_type::op_assign(other, get_empty_vtable());
            return *this;
        }

        polymorphic_executor& operator=(polymorphic_executor&& other) noexcept
        {
            base_type::op_assign(std::move(other), get_empty_vtable());
            return *this;
        }

        template <typename Exec, typename PE = typename std::decay<Exec>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<PE, polymorphic_executor>::value>::type>
        polymorphic_executor(Exec&& exec)
          : base_type(get_empty_vtable())
        {
            assign(std::forward<Exec>(exec));
        }

        template <typename Exec, typename PE = typename std::decay<Exec>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<PE, polymorphic_executor>::value>::type>
        polymorphic_executor& operator=(Exec&& exec)
        {
            assign(std::forward<Exec>(exec));
            return *this;
        }

    private:
        void assign(std::nullptr_t) noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        template <typename Exec>
        void assign(Exec&& exec)
        {
            using T = typename std::decay<Exec>::type;
            static_assert(std::is_constructible<T, T const&>::value,
                "Exec shall be CopyConstructible");

            if (!detail::is_empty_polymorphic_executor(exec))
            {
                vtable const* exec_vptr = get_vtable<T>();
                void* buffer = nullptr;
                if (vptr == exec_vptr)
                {
                    // reuse object storage
                    HPX_ASSERT(object != nullptr);
                    buffer = object;
                    vtable::template get<T>(object).~T();
                }
                else
                {
                    destroy();
                    vptr = exec_vptr;
                    buffer = vtable::template allocate<T>(
                        storage, detail::polymorphic_executor_storage_size);
                }
                object = ::new (buffer) T(std::forward<Exec>(exec));
            }
            else
            {
                base_type::reset(get_empty_vtable());
            }
        }

    public:
        void reset() noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        using base_type::empty;
        using base_type::swap;
        using base_type::operator bool;

        ///////////////////////////////////////////////////////////////////////
        // actual executor interface

        // This executor always exposes future<R>
        template <typename T>
        using future_type = hpx::future<R>;

        // NonBlockingOneWayExecutor interface
        template <typename F>
        HPX_FORCEINLINE void post(F&& f, Ts... ts) const
        {
            using function_type = typename vtable::post_function_type;

            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            vptr->post(object, function_type(std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        // OneWayExecutor interface
        template <typename F>
        HPX_FORCEINLINE R sync_execute(F&& f, Ts... ts) const
        {
            using function_type = typename vtable::sync_execute_function_type;

            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->sync_execute(object, function_type(std::forward<F>(f)),
                std::forward<Ts>(ts)...);
        }

        // TwoWayExecutor interface
        template <typename F>
        HPX_FORCEINLINE hpx::future<R> async_execute(F&& f, Ts... ts) const
        {
            using function_type = typename vtable::async_execute_function_type;

            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->async_execute(object,
                function_type(std::forward<F>(f)), std::forward<Ts>(ts)...);
        }

        template <typename F, typename Future>
        HPX_FORCEINLINE hpx::future<R> then_execute(
            F&& f, Future&& predecessor, Ts&&... ts) const
        {
            using function_type = typename vtable::then_execute_function_type;

            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->then_execute(object, function_type(std::forward<F>(f)),
                hpx::make_shared_future(std::forward<Future>(predecessor)),
                std::forward<Ts>(ts)...);
        }

        // BulkOneWayExecutor interface
        template <typename F, typename Shape>
        HPX_FORCEINLINE std::vector<R> bulk_sync_execute(
            F&& f, Shape const& s, Ts&&... ts) const
        {
            using function_type =
                typename vtable::bulk_sync_execute_function_type;

            detail::range_proxy shape(s);
            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->bulk_sync_execute(object,
                function_type(std::forward<F>(f)), shape,
                std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename Shape>
        HPX_FORCEINLINE std::vector<hpx::future<R>> bulk_async_execute(
            F&& f, Shape const& s, Ts&&... ts) const
        {
            using function_type =
                typename vtable::bulk_async_execute_function_type;

            detail::range_proxy shape(s);
            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->bulk_async_execute(object,
                function_type(std::forward<F>(f)), shape,
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape>
        HPX_FORCEINLINE hpx::future<std::vector<R>> bulk_then_execute(F&& f,
            Shape const& s, hpx::shared_future<void> const& predecessor,
            Ts&&... ts) const
        {
            using function_type =
                typename vtable::bulk_then_execute_function_type;

            detail::range_proxy shape(s);
            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->bulk_then_execute(object,
                function_type(std::forward<F>(f)), shape, predecessor,
                std::forward<Ts>(ts)...);
        }

    private:
        static constexpr vtable const* get_empty_vtable() noexcept
        {
            return detail::get_empty_polymorphic_executor_vtable<R(Ts...)>();
        }

        template <typename T>
        static constexpr vtable const* get_vtable() noexcept
        {
            return detail::get_polymorphic_executor_vtable<vtable, T>();
        }

    protected:
        using base_type::object;
        using base_type::storage;
        using base_type::vptr;
    };

}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <typename Sig>
    struct is_never_blocking_one_way_executor<
        parallel::execution::polymorphic_executor<Sig>> : std::true_type
    {
    };

    template <typename Sig>
    struct is_one_way_executor<parallel::execution::polymorphic_executor<Sig>>
      : std::true_type
    {
    };

    template <typename Sig>
    struct is_two_way_executor<parallel::execution::polymorphic_executor<Sig>>
      : std::true_type
    {
    };

    template <typename Sig>
    struct is_bulk_one_way_executor<
        parallel::execution::polymorphic_executor<Sig>> : std::true_type
    {
    };

    template <typename Sig>
    struct is_bulk_two_way_executor<
        parallel::execution::polymorphic_executor<Sig>> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
