//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_THREADS_THREAD_APR_10_2012_0145PM)
#define HPX_THREADS_THREAD_APR_10_2012_0145PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/date_time_chrono.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <boost/chrono/chrono.hpp>
#include <boost/move/move.hpp>
#include <boost/thread/thread.hpp>

#include <iosfwd>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT thread
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        class id;
        typedef threads::thread_id_type native_handle_type;

        thread() BOOST_NOEXCEPT;

        template <typename F>
        explicit thread(BOOST_FWD_REF(F) f)
          : id_(invalid_thread_id)
        {
            start_thread(boost::forward<F>(f));
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
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT                                         \
          , <hpx/runtime/threads/thread.hpp>))                                \
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
            return invalid_thread_id != id_;
        }

        void join();
        void detach()
        {
            mutex_type::scoped_lock l(mtx_);
            id_ = invalid_thread_id;
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

        // compatibility with older boost thread interface
        static void yield() BOOST_NOEXCEPT;
        static void sleep(boost::posix_time::ptime const& xt);

    private:
        void start_thread(BOOST_RV_REF(HPX_STD_FUNCTION<void()>) func);

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
        thread_id_type id_;

        friend bool operator== (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator!= (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator< (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator> (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator<= (thread::id x, thread::id y) BOOST_NOEXCEPT;
        friend bool operator>= (thread::id x, thread::id y) BOOST_NOEXCEPT;

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>&
        operator<< (std::basic_ostream<Char, Traits>&, thread::id);

    public:
        id() BOOST_NOEXCEPT : id_(invalid_thread_id) {}
        explicit id(thread_id_type i) BOOST_NOEXCEPT : id_(i) {}
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
        HPX_EXPORT thread::id get_id() BOOST_NOEXCEPT;

        HPX_EXPORT void yield() BOOST_NOEXCEPT;

        template <typename Clock, typename Duration>
        void sleep_until(boost::chrono::time_point<Clock, Duration> const& at)
        {
            threads::this_thread::suspend(util::to_ptime(at));
        }

        template <typename Rep, typename Period>
        void sleep_for(boost::chrono::duration<Rep, Period> const& p)
        {
            threads::this_thread::suspend(util::to_time_duration(p));
        }
    }
}}

#endif

#else // BOOST_PP_IS_ITERATING
#define N BOOST_PP_ITERATION()

#define HPX_FWD_ARGS(z, n, _)                                                 \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/

#define HPX_FORWARD_ARGS(z, n, _)                                             \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

    template <typename F, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    thread(BOOST_FWD_REF(F) f, BOOST_PP_ENUM(N, HPX_FWD_ARGS, _))
      : id_(invalid_thread_id)
    {
        start_thread(HPX_STD_BIND(boost::forward<F>(f),
            BOOST_PP_ENUM(N, HPX_FORWARD_ARGS, _)));
    }

#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef N
#endif
