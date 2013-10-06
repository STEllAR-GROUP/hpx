//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM)
#define HPX_LCOS_FUTURE_WAIT_OCT_23_2008_1140AM

#include <hpx/hpx_fwd.hpp>

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <vector>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/move/move.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        // This version has a callback to be invoked for each future when it
        // gets ready.
        template <typename T, typename F>
        struct when_all
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_all)

        protected:
            void on_future_ready_(threads::thread_id_type id)
            {
                std::size_t oldcount = ready_count_.fetch_add(1);
                BOOST_ASSERT(oldcount < lazy_values_.size());

                if (oldcount + 1 == lazy_values_.size())
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type id,
                boost::mpl::false_)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i, lazy_values_[i].get());
                }

                // keep track of ready futures
                on_future_ready_(id);
            }

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type id,
                boost::mpl::true_)
            {
                if (lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    // invoke callback function
                    f_(i);
                }

                // keep track of ready futures
                on_future_ready_(id);
            }

            void on_future_ready(std::size_t i, threads::thread_id_type id)
            {
                on_future_ready_(i, id, boost::is_same<void, T>());
            }

        public:
            typedef std::vector<lcos::future<T> > argument_type;
            typedef std::vector<lcos::future<T> > result_type;

            template <typename F_>
            when_all(argument_type const& lazy_values, BOOST_FWD_REF(F_) f,
                    boost::atomic<std::size_t>* success_counter)
              : lazy_values_(lazy_values),
                ready_count_(0),
                f_(boost::forward<F>(f)),
                success_counter_(success_counter)
            {}

            template <typename F_>
            when_all(BOOST_RV_REF(argument_type) lazy_values, BOOST_FWD_REF(F_) f,
                    boost::atomic<std::size_t>* success_counter)
              : lazy_values_(boost::move(lazy_values)),
                ready_count_(0),
                f_(boost::forward<F>(f)),
                success_counter_(success_counter)
            {}

            when_all(BOOST_RV_REF(when_all) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                ready_count_(rhs.ready_count_.load()),
                f_(boost::move(rhs.f_)),
                success_counter_(rhs.success_counter_)
            {
                rhs.success_counter_ = 0;
            }

            when_all& operator= (BOOST_RV_REF(when_all) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    ready_count_ = rhs.ready_count_;
                    rhs.ready_count_ = 0;
                    f_ = boost::move(rhs.f_);
                    success_counter_ = rhs.success_counter_;
                    rhs.success_counter_ = 0;
                }
                return *this;
            }

            result_type operator()()
            {
                using lcos::detail::get_future_data;

                ready_count_.store(0);

                // set callback functions to executed when future is ready
                std::size_t size = lazy_values_.size();
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    lcos::detail::future_data_base<T>* current =
                        get_future_data(lazy_values_[i]);

                    current->set_on_completed(
                        util::bind(&when_all::on_future_ready, this, i, id));
                }

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (ready_count_ < size)
                {
                    // wait for all of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // all futures should be ready
                BOOST_ASSERT(ready_count_ == size);

                return lazy_values_;
            }

            std::vector<lcos::future<T> > lazy_values_;
            boost::atomic<std::size_t> ready_count_;
            typename util::remove_reference<F>::type f_;
            boost::atomic<std::size_t>* success_counter_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // Asynchronous versions.

    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1, typename F>
    inline std::size_t
    wait (lcos::future<T1> const& f1, F const& f)
    {
        f(0, f1.get());
        return 1;
    }

    template <typename F>
    inline std::size_t
    wait (lcos::future<void> const& f1, F const& f)
    {
        f1.get();
        f(0);
        return 1;
    }

    //////////////////////////////////////////////////////////////////////////
    // This overload of wait() will make sure that the passed function will be
    // invoked as soon as a value becomes available, it will not wait for all
    // results to be there.
    template <typename T1, typename F>
    inline std::size_t
    wait (std::vector<lcos::future<T1> > const& lazy_values, BOOST_FWD_REF(F) f,
        boost::int32_t suspend_for = 10)
    {
        typedef std::vector<lcos::future<T1> > return_type;

        if (lazy_values.empty())
            return 0;

        boost::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::when_all<T1, F>(
                    lazy_values, boost::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }

    template <typename F>
    inline std::size_t
    wait (std::vector<lcos::future<void> > const& lazy_values, BOOST_FWD_REF(F) f,
        boost::int32_t suspend_for = 10)
    {
        typedef std::vector<lcos::future<void> > return_type;

        if (lazy_values.empty())
            return 0;

        boost::atomic<std::size_t> success_counter(0);
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::when_all<void, F>(
                    lazy_values, boost::forward<F>(f), &success_counter));

        p.apply();
        p.get_future().get();

        return success_counter.load();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Synchronous versions.

    /// The one argument version is special in the sense that it returns the
    /// expected value directly (without wrapping it into a tuple).
    template <typename T1>
    inline T1
    wait (lcos::future<T1> const& f1)
    {
        return f1.get();
    }

    inline void
    wait (lcos::future<void> const& f1)
    {
        f1.get();
    }

    template <typename T1, typename T2>
    inline HPX_STD_TUPLE<T1, T2>
    wait (lcos::future<T1> const& f1, lcos::future<T2> const& f2)
    {
        return HPX_STD_MAKE_TUPLE(f1.get(), f2.get());
    }

    inline void
    wait (lcos::future<void> const& f1, lcos::future<void> const& f2)
    {
        f1.get();
        f2.get();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/future_wait.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/future_wait_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_WAIT_ARGUMENT_LIMIT,                                          \
    "hpx/lcos/future_wait.hpp"))                                              \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline void
    wait (std::vector<lcos::future<T> > const& v, std::vector<T>& r)
    {
        r.reserve(v.size());

        typedef lcos::future<T> value_type;
        BOOST_FOREACH(value_type const& f, v)
            r.push_back(f.get());
    }

    template <typename T>
    inline void
    wait (std::vector<lcos::future<T> > const& v)
    {
        typedef lcos::future<T> value_type;
        BOOST_FOREACH(value_type const& f, v)
            f.get();
    }

    inline void
    wait (std::vector<lcos::future<void> > const& v)
    {
        typedef lcos::future<void> value_type;
        BOOST_FOREACH(value_type const& f, v)
            f.get();
    }
}}

namespace hpx
{
    using lcos::wait;
}

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

#define HPX_FUTURE_WAIT_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)             \
        lcos::future<BOOST_PP_CAT(T, n)> const& BOOST_PP_CAT(f, n)            \
    /**/
#define HPX_FUTURE_WAIT_VOID_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)        \
        lcos::future<void> const& BOOST_PP_CAT(f, n)                          \
    /**/
#define HPX_FUTURE_TUPLE_ARGUMENT(z, n, data) BOOST_PP_COMMA_IF(n)            \
        BOOST_PP_CAT(f, n).get()                                              \
    /**/
#define HPX_FUTURE_VOID_STATEMENT(z, n, data) BOOST_PP_CAT(f, n).get();

namespace hpx { namespace lcos
{
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    inline HPX_STD_TUPLE<BOOST_PP_ENUM_PARAMS(N, T)>
    wait (BOOST_PP_REPEAT(N, HPX_FUTURE_WAIT_ARGUMENT, _))
    {
        return HPX_STD_MAKE_TUPLE(BOOST_PP_REPEAT(N, HPX_FUTURE_TUPLE_ARGUMENT, _));
    }

    inline void
    wait (BOOST_PP_REPEAT(N, HPX_FUTURE_WAIT_VOID_ARGUMENT, _))
    {
        BOOST_PP_REPEAT(N, HPX_FUTURE_VOID_STATEMENT, _)
    }
}}

#undef HPX_FUTURE_WAIT_ARGUMENT
#undef HPX_FUTURE_WAIT_VOID_ARGUMENT
#undef HPX_FUTURE_TUPLE_ARGUMENT
#undef HPX_FUTURE_VOID_STATEMENT
#undef N

#endif

