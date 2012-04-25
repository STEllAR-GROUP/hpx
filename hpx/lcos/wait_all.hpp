//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM)
#define HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <vector>

#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        template <typename T, typename RT>
        struct wait_all
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(wait_all)

            void on_future_ready(threads::thread_id_type id)
            {
                mutex_type::scoped_lock l(mtx_);
                if (ready_count_ != lazy_values_.size() &&
                    ++ready_count_ == lazy_values_.size())
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self().get_thread_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef lcos::local::spinlock mutex_type;

            typedef std::vector<lcos::future<T, RT> > argument_type;
            typedef std::vector<lcos::future<T, RT> > result_type;

            wait_all(argument_type const& lazy_values)
              : lazy_values_(lazy_values),
                ready_count_(0)
            {}

            wait_all(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)),
                ready_count_(0)
            {}

            wait_all(BOOST_RV_REF(wait_all) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                ready_count_(rhs.ready_count_),
                mtx_(boost::move(rhs.mtx_))
            {
                rhs.ready_count_ = 0;
            }

            wait_all& operator= (BOOST_RV_REF(wait_all) rhs)
            {
                if (this != &rhs) {
                    mutex_type::scoped_lock l1(mtx_);
                    mutex_type::scoped_lock l2(rhs.mtx_);
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    ready_count_ = rhs.ready_count_;
                    rhs.ready_count_ = 0;
                    mtx_ = boost::move(rhs.mtx_);
                }
                return *this;
            }

            result_type operator()()
            {
                mutex_type::scoped_lock l(mtx_);
                ready_count_ = 0;

                {
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

                    // set callback functions to executed when future is ready
                    threads::thread_id_type id = threads::get_self().get_thread_id();
                    for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                    {
                        lazy_values_[i].when(
                            util::bind(&wait_all::on_future_ready, this, id)
                        );
                    }
                }

                // if all of the requested futures are already set, our
                // callback above has already been called, otherwise we suspend
                // ourselves
                if (ready_count_ != lazy_values_.size())
                {
                    // wait for any of the futures to return to become ready
                    util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended);
                }

                // all futures should be ready
                BOOST_ASSERT(ready_count_ == lazy_values_.size());

              // reset all pending callback functions
              l.unlock();
              for (std::size_t i = 0; i < lazy_values_.size(); ++i)
                  lazy_values_[i].when();

                return lazy_values_;
            }

            std::vector<lcos::future<T, RT> > lazy_values_;
            std::size_t ready_count_;
            mutable mutex_type mtx_;
        };
    }

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \return   The returned future holds the same list of futures as has
    ///           been passed to wait_all.
    template <typename T, typename RT>
    lcos::future<std::vector<lcos::future<T, RT> > >
    wait_all (BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T, RT> >))) lazy_values)
    {
        typedef std::vector<lcos::future<T, RT> > return_type;
        lcos::local::packaged_task<return_type()> p(
            detail::wait_all<T, RT>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }

    template <typename T, typename RT>
    lcos::future<std::vector<lcos::future<T, RT> > >
    wait_all (std::vector<lcos::future<T, RT> > const& lazy_values)
    {
        typedef std::vector<lcos::future<T, RT> > return_type;
        lcos::local::packaged_task<return_type()> p =
            lcos::local::packaged_task<return_type()>(
                detail::wait_all<T, RT>(lazy_values));
        p.apply();
        return p.get_future();
    }
}

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_FUNCTION_LIMIT, <hpx/lcos/wait_all.hpp>))                     \
/**/
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

#define HPX_WAIT_ALL_PUSH_BACK_ARGS(z, n, _)                                  \
        lazy_values.push_back(BOOST_PP_CAT(f, n));                            \
    /**/
#define HPX_WAIT_ALL_FUTURE_ARGS(z, n, _)                                     \
        lcos::future<T, RT> BOOST_PP_CAT(f, n)                                \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RT>
    lcos::future<std::vector<lcos::future<T, RT> > >
    wait_all (BOOST_PP_ENUM(N, HPX_WAIT_ALL_FUTURE_ARGS, _))
    {
        std::vector<lcos::future<T, RT> > lazy_values;
        lazy_values.reserve(N);
        BOOST_PP_REPEAT(N, HPX_WAIT_ALL_PUSH_BACK_ARGS, _)

        typedef std::vector<lcos::future<T, RT> > return_type;
        lcos::local::packaged_task<return_type()> p(
            detail::wait_all<T, RT>(boost::move(lazy_values)));
        p.apply();
        return p.get_future();
    }
}

#undef HPX_WAIT_ALL_FUTURE_ARGS
#undef HPX_WAIT_ALL_PUSH_BACK_ARGS
#undef N

#endif

