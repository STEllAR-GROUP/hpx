//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM)
#define HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/move.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <vector>

#include <boost/assert.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <boost/atomic.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        //////////////////////////////////////////////////////////////////////
        template <typename T, typename F = void>
        struct when_all;

        //////////////////////////////////////////////////////////////////////
        // This version has an additional callback to be invoked for each 
        // future when it gets ready.
        template <typename T, typename F>
        struct when_all : when_all<T, void>
        {
        private:
            typedef when_all<T, void> base_type;

            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_all)

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type id, boost::mpl::false_)
            {
                if (this->base_type::lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    f_(i, this->base_type::lazy_values_[i].get());  // invoke callback function
                }
                this->base_type::on_future_ready(id);   // keep track of ready futures
            }

            template <typename Index>
            void on_future_ready_(Index i, threads::thread_id_type id, boost::mpl::true_)
            {
                if (this->base_type::lazy_values_[i].has_value()) {
                    if (success_counter_)
                        ++*success_counter_;
                    f_(i);                              // invoke callback function
                }
                this->base_type::on_future_ready(id);   // keep track of ready futures
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
              : base_type(lazy_values), 
                f_(boost::forward<F>(f)),
                success_counter_(success_counter)
            {}

            template <typename F_>
            when_all(BOOST_RV_REF(argument_type) lazy_values, BOOST_FWD_REF(F_) f,
                    boost::atomic<std::size_t>* success_counter)
              : base_type(boost::move(lazy_values)), 
                f_(boost::forward<F>(f)),
                success_counter_(success_counter)
            {}

            when_all(BOOST_RV_REF(when_all) rhs)
              : base_type(boost::move(rhs.lazy_values_)), 
                f_(boost::move(rhs.f_)),
                success_counter_(rhs.success_counter_)
            {
                rhs.success_counter_ = 0;
            }

            when_all& operator= (BOOST_RV_REF(when_all) rhs)
            {
                if (this != &rhs) {
                    this->base_type::operator=(boost::move(rhs));
                    f_ = boost::move(rhs.f_);
                    success_counter_ = rhs.success_counter_;
                    rhs.success_counter_ = 0;
                }
                return *this;
            }

            result_type operator()()
            {
                using lcos::detail::get_future_data;
                this->base_type::ready_count_.store(0);

                // set callback functions to executed when future is ready
                std::size_t size = this->base_type::lazy_values_.size();
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    get_future_data(this->base_type::lazy_values_[i])->set_on_completed(
                        util::bind(&when_all::on_future_ready, this, i, id));
                }

                // If all of the requested futures are already set then our
                // callback above has already been called, otherwise we suspend
                // ourselves.
                if (this->base_type::ready_count_ < size)
                {
                    // wait for all of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // all futures should be ready
                BOOST_ASSERT(this->base_type::ready_count_ == size);

                // reset all pending callback functions
                for (std::size_t i = 0; i < size; ++i) 
                    get_future_data(this->base_type::lazy_values_[i])->reset_on_completed();

                return this->base_type::lazy_values_;
            }

            typename std::remove_reference<F>::type f_;
            boost::atomic<std::size_t>* success_counter_;
        };

        //////////////////////////////////////////////////////////////////////
        template <typename T>
        struct when_all<T, void>
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_all)

        protected:
            void on_future_ready(threads::thread_id_type id)
            {
                if (ready_count_.fetch_add(1) + 1 == lazy_values_.size())
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef std::vector<lcos::future<T> > argument_type;
            typedef std::vector<lcos::future<T> > result_type;

            when_all(argument_type const& lazy_values)
              : lazy_values_(lazy_values),
                ready_count_(0)
            {}

            when_all(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values)),
                ready_count_(0)
            {}

            when_all(BOOST_RV_REF(when_all) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_)),
                ready_count_(rhs.ready_count_.load())
            {
                rhs.ready_count_.store(0);
            }

            when_all& operator= (BOOST_RV_REF(when_all) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);
                    ready_count_ = rhs.ready_count_;
                    rhs.ready_count_ = 0;
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
                    get_future_data(lazy_values_[i])->set_on_completed(
                        util::bind(&when_all::on_future_ready, this, id));
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

                // reset all pending callback functions
                for (std::size_t i = 0; i < size; ++i) 
                    get_future_data(lazy_values_[i])->reset_on_completed();

                return lazy_values_;
            }

            std::vector<lcos::future<T> > lazy_values_;
            boost::atomic<std::size_t> ready_count_;
        };
    }

    /// The function \a when_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \return   The returned future holds the same list of futures as has
    ///           been passed to when_all.

    template <typename Iterator>
    lcos::future<std::vector<lcos::future<
        typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
    > > >
    when_all(Iterator begin, Iterator end)
    {
        typedef typename lcos::future_iterator_traits<
            Iterator>::traits_type::value_type value_type;
        typedef std::vector<lcos::future<value_type> > return_type;

        return_type lazy_values;
        std::copy(begin, end, std::back_inserter(lazy_values));

        if (lazy_values.empty())
            return lcos::make_future(return_type());

        lcos::local::futures_factory<return_type()> p(
            detail::when_all<value_type>(boost::move(lazy_values)));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values)
    {
        typedef std::vector<lcos::future<T> > return_type;

        if (lazy_values.empty())
            return lcos::make_future(return_type());

        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all(std::vector<lcos::future<T> > const& lazy_values)
    {
        typedef std::vector<lcos::future<T> > return_type;

        if (lazy_values.empty())
            return lcos::make_future(return_type());

        lcos::local::futures_factory<return_type()> p =
            lcos::local::futures_factory<return_type()>(
                detail::when_all<T>(lazy_values));

        p.apply();
        return p.get_future();
    }

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \a wait_all returns after all futures have been triggered.

    template <typename Iterator>
    std::vector<
        typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
    >
    wait_all(Iterator begin, Iterator end)
    {
        typedef std::vector<
            typename lcos::future_iterator_traits<Iterator>::traits_type::value_type
        > result_type;

        lcos::future<result_type> f = when_all(begin, end);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get();
    }

    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<T> >))) lazy_values)
    {
        typedef std::vector<lcos::future<T> > result_type;

        lcos::future<result_type> f = when_all(lazy_values);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get();
    }

    template <typename T>
    std::vector<lcos::future<T> >
    wait_all (std::vector<lcos::future<T> > const& lazy_values)
    {
        typedef std::vector<lcos::future<T> > result_type;

        lcos::future<result_type> f = when_all(lazy_values);
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get();
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_all.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_all_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_all.hpp>))                \
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

#define HPX_WHEN_ALL_PUSH_BACK_ARG(z, n, _)                                   \
        lazy_values.push_back(BOOST_PP_CAT(f, n));                            \
    /**/
#define HPX_WHEN_ALL_FUTURE_ARG(z, n, _)                                      \
        lcos::future<T> BOOST_PP_CAT(f, n)                                    \
    /**/
#define HPX_WHEN_ALL_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    lcos::future<std::vector<lcos::future<T> > >
    when_all (BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_ARG, _))
    {
        typedef std::vector<lcos::future<T> > return_type;

        return_type lazy_values;
        lazy_values.reserve(N);
        BOOST_PP_REPEAT(N, HPX_WHEN_ALL_PUSH_BACK_ARG, _)

        lcos::local::futures_factory<return_type()> p(
            detail::when_all<T>(boost::move(lazy_values)));

        p.apply();
        return p.get_future();
    }

    template <typename T>
    std::vector<lcos::future<T> >
    wait_all(BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_ARG, _))
    {
        typedef std::vector<lcos::future<T> > result_type;

        lcos::future<result_type> f = when_all(
            BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_VAR, _));
        if (!f.valid()) {
            HPX_THROW_EXCEPTION(uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get();
    }
}

#undef HPX_WHEN_ALL_FUTURE_VAR
#undef HPX_WHEN_ALL_FUTURE_ARG
#undef HPX_WHEN_ALL_PUSH_BACK_ARG
#undef N

#endif

