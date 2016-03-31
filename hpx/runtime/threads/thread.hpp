//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#if !defined(HPX_THREADS_THREAD_APR_10_2012_0145PM)
#define HPX_THREADS_THREAD_APR_10_2012_0145PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/thread/locks.hpp>
#include <boost/thread/thread.hpp>
#include <boost/utility/enable_if.hpp>

#include <iosfwd>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT thread
    {
        typedef lcos::local::spinlock mutex_type;
        void terminate(const char * function, const char * reason) const;

    public:
        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() HPX_NOEXCEPT;

        template <typename F>
        explicit thread(F && f)
          : id_(threads::invalid_thread_id)
        {
            start_thread(util::deferred_call(std::forward<F>(f)));
        }

        template <typename F, typename ...Ts>
        explicit thread(F&& f, Ts&&... vs)
        {
            start_thread(util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(vs)...));
        }

        ~thread();

        HPX_MOVABLE_ONLY(thread);

    public:
        thread(thread &&) HPX_NOEXCEPT;
        thread& operator=(thread &&) HPX_NOEXCEPT;

        void swap(thread&) HPX_NOEXCEPT;
        bool joinable() const HPX_NOEXCEPT
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return joinable_locked();
        }

        void join();
        void detach()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            detach_locked();
        }

        id get_id() const HPX_NOEXCEPT;

        native_handle_type native_handle() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return id_;
        }

        static std::size_t hardware_concurrency() HPX_NOEXCEPT;

        // extensions
        void interrupt(bool flag = true);
        bool interruption_requested() const;

        static void interrupt(id, bool flag = true);

        lcos::future<void> get_future(error_code& ec = throws);

#if defined(HPX_HAVE_THREAD_LOCAL_STORAGE)
        std::size_t get_thread_data() const;
        std::size_t set_thread_data(std::size_t);
#endif

    private:
        bool joinable_locked() const HPX_NOEXCEPT
        {
            return threads::invalid_thread_id != id_;
        }
        void detach_locked()
        {
            id_ = threads::invalid_thread_id;
        }
        void start_thread(util::unique_function_nonser<void()> && func);
        static threads::thread_state_enum thread_function_nullary(
            util::unique_function_nonser<void()> const& func);

        mutable mutex_type mtx_;
        threads::thread_id_type id_;
    };

    inline void swap(thread& x, thread& y) HPX_NOEXCEPT
    {
        x.swap(y);
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread::id
    {
    private:
        threads::thread_id_type id_;

        friend bool operator== (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;
        friend bool operator!= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;
        friend bool operator< (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;
        friend bool operator> (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;
        friend bool operator<= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;
        friend bool operator>= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<< (std::basic_ostream<Char, Traits>&, thread::id const&);

        friend class thread;

    public:
        id() HPX_NOEXCEPT : id_(threads::invalid_thread_id) {}
        explicit id(threads::thread_id_type i) HPX_NOEXCEPT : id_(i) {}
    };

    inline bool operator== (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return x.id_.get() == y.id_.get();
    }

    inline bool operator!= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return !(x == y);
    }

    inline bool operator< (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return x.id_.get() < y.id_.get();
    }

    inline bool operator> (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return x.id_.get() > y.id_.get();
    }

    inline bool operator<= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return !(x.id_.get() > y.id_.get());
    }

    inline bool operator>= (thread::id const& x, thread::id const& y) HPX_NOEXCEPT
    {
        return !(x.id_.get() < y.id_.get());
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
        HPX_API_EXPORT thread::id get_id() HPX_NOEXCEPT;

        HPX_API_EXPORT void yield() HPX_NOEXCEPT;

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

#endif
