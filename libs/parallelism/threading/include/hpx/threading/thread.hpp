//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/futures/future_fwd.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading_base/scheduler_base.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <type_traits>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    using thread_termination_handler_type =
        util::function_nonser<void(std::exception_ptr const& e)>;
    HPX_PARALLELISM_EXPORT void set_thread_termination_handler(
        thread_termination_handler_type f);

    class HPX_PARALLELISM_EXPORT thread
    {
        typedef lcos::local::spinlock mutex_type;
        void terminate(const char* function, const char* reason) const;

    public:
        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() noexcept;

        template <typename F,
            typename Enable = typename std::enable_if<!std::is_same<
                typename std::decay<F>::type, thread>::value>::type>
        explicit thread(F&& f)
        {
            auto thrd_data = threads::get_self_id_data();
            HPX_ASSERT(thrd_data);
            start_thread(thrd_data->get_scheduler_base()->get_parent_pool(),
                util::deferred_call(std::forward<F>(f)));
        }

        template <typename F, typename... Ts>
        explicit thread(F&& f, Ts&&... vs)
        {
            auto thrd_data = threads::get_self_id_data();
            HPX_ASSERT(thrd_data);
            start_thread(thrd_data->get_scheduler_base()->get_parent_pool(),
                util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(vs)...));
        }

        template <typename F>
        thread(threads::thread_pool_base* pool, F&& f)
        {
            start_thread(pool, util::deferred_call(std::forward<F>(f)));
        }

        template <typename F, typename... Ts>
        thread(threads::thread_pool_base* pool, F&& f, Ts&&... vs)
        {
            start_thread(pool,
                util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(vs)...));
        }

        ~thread();

    public:
        thread(thread&&) noexcept;
        thread& operator=(thread&&) noexcept;

        void swap(thread&) noexcept;
        bool joinable() const noexcept
        {
            std::lock_guard<mutex_type> l(mtx_);
            return joinable_locked();
        }

        void join();
        void detach()
        {
            std::lock_guard<mutex_type> l(mtx_);
            detach_locked();
        }

        id get_id() const noexcept;

        native_handle_type native_handle() const    //-V659
        {
            std::lock_guard<mutex_type> l(mtx_);
            return id_;
        }

        HPX_NODISCARD static unsigned int hardware_concurrency() noexcept;

        // extensions
        void interrupt(bool flag = true);
        bool interruption_requested() const;

        static void interrupt(id, bool flag = true);

        lcos::future<void> get_future(error_code& ec = throws);

        std::size_t get_thread_data() const;
        std::size_t set_thread_data(std::size_t);

#if defined(HPX_HAVE_LIBCDS)
        std::size_t get_libcds_data() const;
        std::size_t set_libcds_data(std::size_t);
        std::size_t get_libcds_hazard_pointer_data() const;
        std::size_t set_libcds_hazard_pointer_data(std::size_t);
        std::size_t get_libcds_dynamic_hazard_pointer_data() const;
        std::size_t set_libcds_dynamic_hazard_pointer_data(std::size_t);
#endif

    private:
        bool joinable_locked() const noexcept
        {
            return threads::invalid_thread_id != id_;
        }
        void detach_locked()
        {
            id_ = threads::invalid_thread_id;
        }
        void start_thread(threads::thread_pool_base* pool,
            util::unique_function_nonser<void()>&& func);
        static threads::thread_result_type thread_function_nullary(
            util::unique_function_nonser<void()> const& func);

        mutable mutex_type mtx_;
        threads::thread_id_type id_;
    };

    inline void swap(thread& x, thread& y) noexcept
    {
        x.swap(y);
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread::id
    {
    private:
        threads::thread_id_type id_;

        friend bool operator==(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator!=(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator<(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator>(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator<=(
            thread::id const& x, thread::id const& y) noexcept;
        friend bool operator>=(
            thread::id const& x, thread::id const& y) noexcept;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>& operator<<(
            std::basic_ostream<Char, Traits>&, thread::id const&);

        friend class thread;

    public:
        id() noexcept
          : id_(threads::invalid_thread_id)
        {
        }
        explicit id(threads::thread_id_type const& i) noexcept
          : id_(i)
        {
        }
        explicit id(threads::thread_id_type&& i) noexcept
          : id_(std::move(i))
        {
        }

        threads::thread_id_type const& native_handle() const
        {
            return id_;
        }
    };

    inline bool operator==(thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ == y.id_;
    }

    inline bool operator!=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x == y);
    }

    inline bool operator<(thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ < y.id_;
    }

    inline bool operator>(thread::id const& x, thread::id const& y) noexcept
    {
        return y < x;
    }

    inline bool operator<=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x > y);
    }

    inline bool operator>=(thread::id const& x, thread::id const& y) noexcept
    {
        return !(x < y);
    }

    template <typename Char, typename Traits>
    std::basic_ostream<Char, Traits>& operator<<(
        std::basic_ostream<Char, Traits>& out, thread::id const& id)
    {
        out << id.id_;
        return out;
    }

    //     template <class T> struct hash;
    //     template <> struct hash<thread::id>;

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread {
        HPX_PARALLELISM_EXPORT thread::id get_id() noexcept;

        HPX_PARALLELISM_EXPORT void yield() noexcept;
        HPX_PARALLELISM_EXPORT void yield_to(thread::id) noexcept;

        // extensions
        HPX_PARALLELISM_EXPORT threads::thread_priority get_priority();
        HPX_PARALLELISM_EXPORT std::ptrdiff_t get_stack_size();

        HPX_PARALLELISM_EXPORT void interruption_point();
        HPX_PARALLELISM_EXPORT bool interruption_enabled();
        HPX_PARALLELISM_EXPORT bool interruption_requested();

        HPX_PARALLELISM_EXPORT void interrupt();

        HPX_PARALLELISM_EXPORT void sleep_until(
            hpx::chrono::steady_time_point const& abs_time);

        inline void sleep_for(hpx::chrono::steady_duration const& rel_time)
        {
            sleep_until(rel_time.from_now());
        }

        HPX_PARALLELISM_EXPORT std::size_t get_thread_data();
        HPX_PARALLELISM_EXPORT std::size_t set_thread_data(std::size_t);

#if defined(HPX_HAVE_LIBCDS)
        HPX_PARALLELISM_EXPORT std::size_t get_libcds_data();
        HPX_PARALLELISM_EXPORT std::size_t set_libcds_data(std::size_t);
        HPX_PARALLELISM_EXPORT std::size_t get_libcds_hazard_pointer_data();
        HPX_PARALLELISM_EXPORT std::size_t set_libcds_hazard_pointer_data(
            std::size_t);
        HPX_PARALLELISM_EXPORT std::size_t
        get_libcds_dynamic_hazard_pointer_data();
        HPX_PARALLELISM_EXPORT std::size_t
            set_libcds_dynamic_hazard_pointer_data(std::size_t);
#endif

        class HPX_PARALLELISM_EXPORT disable_interruption
        {
        private:
            disable_interruption(disable_interruption const&);
            disable_interruption& operator=(disable_interruption const&);

            bool interruption_was_enabled_;
            friend class restore_interruption;

        public:
            disable_interruption();
            ~disable_interruption();
        };

        class HPX_PARALLELISM_EXPORT restore_interruption
        {
        private:
            restore_interruption(restore_interruption const&);
            restore_interruption& operator=(restore_interruption const&);

            bool interruption_was_enabled_;

        public:
            explicit restore_interruption(disable_interruption& d);
            ~restore_interruption();
        };
    }    // namespace this_thread
}    // namespace hpx

namespace std {

    // specialize std::hash for hpx::thread::id
    template <>
    struct hash<::hpx::thread::id>
    {
        std::size_t operator()(::hpx::thread::id const& id) const
        {
            std::hash<::hpx::threads::thread_id_type> hasher_;
            return hasher_(id.native_handle());
        }
    };
}    // namespace std

#include <hpx/config/warnings_suffix.hpp>
