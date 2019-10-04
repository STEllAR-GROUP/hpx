//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_THREADS_THREAD_HPP
#define HPX_RUNTIME_THREADS_THREAD_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_pool_base.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/util_fwd.hpp>

#include <cstddef>
#include <exception>
#include <iosfwd>
#include <mutex>
#include <utility>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    using thread_termination_handler_type =
        std::function<void(std::exception_ptr const& e)>;
    void set_thread_termination_handler(thread_termination_handler_type f);

    class thread
    {
        typedef lcos::local::spinlock mutex_type;
        void terminate(const char* function, const char* reason) const;

    public:
        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() noexcept;

        template <typename F, typename Enable = typename
            std::enable_if<!std::is_same<typename hpx::util::decay<F>::type,
                thread>::value>::type>
        explicit thread(F&& f)
        {
            HPX_ASSERT(threads::get_self_ptr());
            start_thread(
                threads::get_self_id()->get_scheduler_base()->get_parent_pool(),
                util::deferred_call(std::forward<F>(f)));
        }

        template <typename F, typename... Ts>
        explicit thread(F&& f, Ts&&... vs)
        {
            HPX_ASSERT(threads::get_self_ptr());
            start_thread(
                threads::get_self_id()->get_scheduler_base()->get_parent_pool(),
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

        native_handle_type native_handle() const //-V659
        {
            std::lock_guard<mutex_type> l(mtx_);
            return id_;
        }

        static std::size_t hardware_concurrency() noexcept;

        // extensions
        void interrupt(bool flag = true);
        bool interruption_requested() const;

        static void interrupt(id, bool flag = true);

        lcos::future<void> get_future(error_code& ec = throws);

        std::size_t get_thread_data() const;
        std::size_t set_thread_data(std::size_t);

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

        friend bool operator== (thread::id const& x, thread::id const& y) noexcept;
        friend bool operator!= (thread::id const& x, thread::id const& y) noexcept;
        friend bool operator< (thread::id const& x, thread::id const& y) noexcept;
        friend bool operator> (thread::id const& x, thread::id const& y) noexcept;
        friend bool operator<= (thread::id const& x, thread::id const& y) noexcept;
        friend bool operator>= (thread::id const& x, thread::id const& y) noexcept;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<< (std::basic_ostream<Char, Traits>&, thread::id const&);

        friend class thread;

    public:
        id() noexcept : id_(threads::invalid_thread_id) {}
        explicit id(threads::thread_id_type const& i) noexcept
          : id_(i)
        {}
        explicit id(threads::thread_id_type && i) noexcept
          : id_(std::move(i))
        {}

        threads::thread_id_type const& native_handle() const { return id_; }
    };

    inline bool operator== (thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ == y.id_;
    }

    inline bool operator!= (thread::id const& x, thread::id const& y) noexcept
    {
        return !(x == y);
    }

    inline bool operator< (thread::id const& x, thread::id const& y) noexcept
    {
        return x.id_ < y.id_;
    }

    inline bool operator> (thread::id const& x, thread::id const& y) noexcept
    {
        return y < x;
    }

    inline bool operator<= (thread::id const& x, thread::id const& y) noexcept
    {
        return !(x > y);
    }

    inline bool operator>= (thread::id const& x, thread::id const& y) noexcept
    {
        return !(x < y);
    }

    template <typename Char, typename Traits>
    std::basic_ostream<Char, Traits>&
    operator<< (std::basic_ostream<Char, Traits>& out, thread::id const& id)
    {
        out << id.id_;
        return out;
    }

//     template <class T> struct hash;
//     template <> struct hash<thread::id>;

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread
    {
        HPX_API_EXPORT thread::id get_id() noexcept;

        HPX_API_EXPORT void yield() noexcept;
        HPX_API_EXPORT void yield_to(thread::id) noexcept;

        // extensions
        HPX_API_EXPORT threads::thread_priority get_priority();
        HPX_API_EXPORT std::ptrdiff_t get_stack_size();

        HPX_API_EXPORT void interruption_point();
        HPX_API_EXPORT bool interruption_enabled();
        HPX_API_EXPORT bool interruption_requested();

        HPX_API_EXPORT void interrupt();

        HPX_API_EXPORT void sleep_until(util::steady_time_point const& abs_time);

        inline void sleep_for(util::steady_duration const& rel_time)
        {
            sleep_until(rel_time.from_now());
        }

        HPX_API_EXPORT std::size_t get_thread_data();
        HPX_API_EXPORT std::size_t set_thread_data(std::size_t);

        class HPX_EXPORT disable_interruption
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

        class HPX_EXPORT restore_interruption
        {
        private:
            restore_interruption(restore_interruption const&);
            restore_interruption& operator=(restore_interruption const&);

            bool interruption_was_enabled_;

        public:
            explicit restore_interruption(disable_interruption& d);
            ~restore_interruption();
        };
    }
}

#include <hpx/config/warnings_suffix.hpp>

#endif /*HPX_RUNTIME_THREADS_THREAD_HPP*/
