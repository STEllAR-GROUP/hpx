//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_THREADS_THREAD_APR_10_2012_0145PM)
#define HPX_THREADS_THREAD_APR_10_2012_0145PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/traits/supports_result_of.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/thread/thread.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <iosfwd>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT thread
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        static threads::thread_id_type const uninitialized;

        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() BOOST_NOEXCEPT;

        template <typename F>
        explicit thread(BOOST_FWD_REF(F) f)
          : id_(uninitialized)
        {
            start_thread(boost::move(HPX_STD_FUNCTION<void()>(boost::forward<F>(f))));
        }

// #if !defined(BOOST_NO_VARIADIC_TEMPLATES)
//         template <typename F, typename ...Args>
//         explicit thread(F&& f, Args&&... args)
//         {
//             start_thead(HPX_STD_BIND(f, boost::forward<Args...>(args)));
//         }
// #else
        // Vertical preprocessor repetition to define the remaining constructors
#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_FUNCTION_ARGUMENT_LIMIT, <hpx/runtime/threads/thread.hpp>))   \
    /**/

#include BOOST_PP_ITERATE()
// #endif

        ~thread();

#if !defined(BOOST_NO_DELETED_FUNCTIONS)
        thread(thread const&) = delete;
        thread& operator=(thread const&) = delete;
#else
    private:
        BOOST_MOVABLE_BUT_NOT_COPYABLE(thread);
#endif

    public:
        thread(BOOST_RV_REF(thread)) BOOST_NOEXCEPT;
        thread& operator=(BOOST_RV_REF(thread)) BOOST_NOEXCEPT;

        void swap(thread&) BOOST_NOEXCEPT;
        bool joinable() const BOOST_NOEXCEPT
        {
            mutex_type::scoped_lock l(mtx_);
            return threads::invalid_thread_id != id_ && uninitialized != id_;
        }

        void join();
        void detach()
        {
            mutex_type::scoped_lock l(mtx_);
            id_ = threads::invalid_thread_id;
        }

        id get_id() const BOOST_NOEXCEPT;

        native_handle_type native_handle() const
        {
            mutex_type::scoped_lock l(mtx_);
            return id_;
        }

        static unsigned hardware_concurrency() BOOST_NOEXCEPT
        {
            return boost::thread::hardware_concurrency();
        }

        // extensions
        void interrupt();
        bool interruption_requested() const;

        static void interrupt(id);

        lcos::future<void> get_future(error_code& ec = throws);

    private:
        void start_thread(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func);
        static threads::thread_state_enum thread_function_nullary(
            HPX_STD_FUNCTION<void()> const& func);

        mutable mutex_type mtx_;
        native_handle_type id_;
    };

    inline void swap(thread& x, thread& y) BOOST_NOEXCEPT
    {
        x.swap(y);
    }

    ///////////////////////////////////////////////////////////////////////////
    class thread::id
    {
    private:
        threads::thread_id_type id_;

        friend bool operator== (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator!= (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator< (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator> (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator<= (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator>= (thread::id x, thread::id y) BOOST_NOEXCEPT;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<< (std::basic_ostream<Char, Traits>&, thread::id);

        friend class thread;

    public:
        id() BOOST_NOEXCEPT : id_(thread::uninitialized) {}
        explicit id(threads::thread_id_type i) BOOST_NOEXCEPT : id_(i) {}
    };

    inline bool operator== (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return x.id_ == y.id_;
    }

    inline bool operator!= (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return !(x == y);
    }

    inline bool operator< (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return x.id_ < y.id_;
    }

    inline bool operator> (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return x.id_ > y.id_;
    }

    inline bool operator<= (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return !(x.id_ > y.id_);
    }

    inline bool operator>= (thread::id x, thread::id y) BOOST_NOEXCEPT
    {
        return !(x.id_ < y.id_);
    }

    template <typename Char, typename Traits>
    std::basic_ostream<Char, Traits>&
    operator<< (std::basic_ostream<Char, Traits>& out, thread::id id)
    {
        out << id.id_;
        return out;
    }

//     template <class T> struct hash;
//     template <> struct hash<thread::id>;

    ///////////////////////////////////////////////////////////////////////////
    namespace this_thread
    {
        HPX_API_EXPORT thread::id get_id() BOOST_NOEXCEPT;

        HPX_API_EXPORT void yield() BOOST_NOEXCEPT;

        // extensions
        HPX_API_EXPORT void interruption_point();
        HPX_API_EXPORT bool interruption_enabled();
        HPX_API_EXPORT bool interruption_requested();

        HPX_API_EXPORT void sleep_until(boost::posix_time::ptime const& at);
        HPX_API_EXPORT void sleep_for(boost::posix_time::time_duration const& p);

        template <typename Clock, typename Duration>
        void sleep_until(boost::chrono::time_point<Clock, Duration> const& at)
        {
            sleep_until(util::to_ptime(at));
        }

        template <typename Rep, typename Period>
        void sleep_for(boost::chrono::duration<Rep, Period> const& p)
        {
            sleep_for(util::to_time_duration(p));
        }

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

        public:
            explicit restore_interruption(disable_interruption& d);
            ~restore_interruption();
        };
    }
}

#include <hpx/config/warnings_suffix.hpp>

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    thread(BOOST_FWD_REF(F) f, HPX_ENUM_FWD_ARGS(N, Arg, arg))
      : id_(uninitialized)
    {
        start_thread(HPX_STD_BIND(boost::forward<F>(f),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
    }

#undef N

#endif

