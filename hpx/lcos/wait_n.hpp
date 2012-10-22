//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_N_APR_19_2012_0203PM)
#define HPX_LCOS_WAIT_N_APR_19_2012_0203PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <vector>

#include <boost/assert.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        template <typename T>
        struct when_n
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_n)

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                mutex_type::scoped_lock l(mtx_);
                if (ready_.size() != needed_count_) {
                    ready_.push_back(idx);
                    if (ready_.size() == needed_count_)
                    {
                        // reactivate waiting thread only if it's not us
                        if (id != threads::get_self().get_thread_id())
                            threads::set_thread_state(id, threads::pending);
                    }
                }
            }

        public:
            typedef lcos::local::spinlock mutex_type;

            typedef std::vector<lcos::future<T> > argument_type;
            typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
                result_type;

            when_n(argument_type const& lazy_values, std::size_t n)
              : lazy_values_(lazy_values),
                needed_count_(n)
            {}

            when_n(BOOST_RV_REF(argument_type) lazy_values, std::size_t n)
              : lazy_values_(boost::move(lazy_values)),
                needed_count_(n)
            {}

            when_n(BOOST_RV_REF(when_n) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
              ready_(boost::move(rhs.ready_)),
                needed_count_(rhs.needed_count_),
                mtx_(boost::move(rhs.mtx_))
            {
                rhs.needed_count_ = 0;
            }

            when_n& operator= (BOOST_RV_REF(when_n) rhs)
            {
                if (this != &rhs) {
                    mutex_type::scoped_lock l1(mtx_);
                    mutex_type::scoped_lock l2(rhs.mtx_);
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    ready_ = boost::move(rhs.ready_);
                    needed_count_ = rhs.needed_count_;
                    rhs.needed_count_ = 0;
                    mtx_ = boost::move(rhs.mtx_);
                }
                return *this;
            }

            result_type operator()()
            {
                mutex_type::scoped_lock l(mtx_);
                ready_.clear();

                {
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

                    // set callback functions to executed when future is ready
                    threads::thread_id_type id = threads::get_self().get_thread_id();
                    for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                    {
                        lazy_values_[i].when(
                            util::bind(&when_n::on_future_ready, this, i, id)
                        );
                    }
                }

                // if all of the requested futures are already set, our
                // callback above has already been called, otherwise we suspend
                // ourselves
                if (ready_.size() != needed_count_)
                {
                    // wait for any of the futures to return to become ready
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended);
                }

                // all futures should be ready
                BOOST_ASSERT(ready_.size() == needed_count_);

                result_type result;
                result.reserve(needed_count_);
                for (std::size_t i = 0; i < needed_count_; ++i) {
                    result.push_back(HPX_STD_MAKE_TUPLE(
                        static_cast<int>(ready_[i]), lazy_values_[ready_[i]]));
                }

                // reset all pending callback functions
                l.unlock();
                for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                    lazy_values_[i].when();

                return result;
            }

            std::vector<lcos::future<T> > lazy_values_;
            std::vector<std::size_t> ready_;
            std::size_t needed_count_;
            mutable mutex_type mtx_;
        };
    }

    /// The function \a when_n is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \return   The returned future holds the same list of futures as has
    ///           been passed to when_n.
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }

    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, std::vector<lcos::future<T> > const& lazy_values)
    {
        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::when_n<T>(lazy_values, n));
        p.apply();
        return p.get_future();
    }

    /// The function \a wait_n is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new vector object representing the list of
    /// the first N futures finished executing.
    ///
    /// \return   The returned vector holds the same list of futures as has
    ///           been passed to wait_n.
    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values)
    {
        return when_n(lazy_values).get();
    }

    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, std::vector<lcos::future<T> > const& lazy_values)
    {
        return when_n(lazy_values).get();
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_n.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_n_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_n.hpp>))                  \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WHEN_N_PUSH_BACK_ARG(z, n, _)                                     \
        lazy_values.push_back(BOOST_PP_CAT(f, n));                            \
    /**/
#define HPX_WHEN_N_FUTURE_ARG(z, n, _)                                        \
        lcos::future<T> BOOST_PP_CAT(f, n)                                    \
    /**/
#define HPX_WHEN_N_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                     \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    lcos::future<std::vector<HPX_STD_TUPLE<int, lcos::future<T> > > >
    when_n(std::size_t n, BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_ARG, _))
    {
        std::vector<lcos::future<T> > lazy_values;
        lazy_values.reserve(N);
        BOOST_PP_REPEAT(N, HPX_WHEN_N_PUSH_BACK_ARG, _)

        typedef std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
            return_type;
        lcos::local::futures_factory<return_type()> p(
            detail::when_n<T>(boost::move(lazy_values), n));
        p.apply();
        return p.get_future();
    }

    template <typename T>
    std::vector<HPX_STD_TUPLE<int, lcos::future<T> > >
    wait_n(std::size_t n, BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_ARG, _))
    {
        return when_n(BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_VAR, _)).get();
    }
}

#undef HPX_WHEN_N_FUTURE_VAR
#undef HPX_WHEN_N_FUTURE_ARG
#undef HPX_WHEN_N_PUSH_BACK_ARG
#undef N

#endif

