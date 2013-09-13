//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
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
#include <hpx/lcos/wait_n.hpp>
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
        // This version has an additional callback to be invoked for each 
        // future when it gets ready.
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
            typename std::remove_reference<F>::type f_;
            boost::atomic<std::size_t>* success_counter_;
        };
    }

    /// The function \a when_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \return   The returned future holds the same list of futures as has
    ///           been passed to when_all.

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > >
    when_all(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > return_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(return_type());

        return when_n(lazy_values.size(), lazy_values, ec);
    }

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > > //-V659
    when_all(std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;
        
        result_type lazy_values_(lazy_values);
        return when_all(boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<lcos::future<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    > > >
    when_all(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef typename lcos::detail::future_iterator_traits<
            Iterator>::traits_type::type value_type;
        typedef std::vector<lcos::future<value_type> > result_type;

        result_type lazy_values_(begin, end);
        return when_all(boost::move(lazy_values_), ec);
    }

    inline lcos::future<HPX_STD_TUPLE<>>
    when_all(error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        return lcos::make_ready_future(result_type());
    }

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \a wait_all returns after all futures have been triggered.

    template <typename R>
    std::vector<lcos::future<R> >
    wait_all(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        lcos::future<result_type> f = when_all(lazy_values, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }

    template <typename R>
    std::vector<lcos::future<R> >
    wait_all(std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        result_type lazy_values_(lazy_values);
        return wait_all(boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    >
    wait_all(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef std::vector<lcos::future<
            typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
        > > result_type;
        
        result_type lazy_values_(begin, end);
        return wait_all(boost::move(lazy_values_), ec);
    }

    inline HPX_STD_TUPLE<>
    wait_all(error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        lcos::future<result_type> f = when_all(ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
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

#define HPX_WHEN_ALL_FUTURE_TYPE(z, n, _)                                     \
        lcos::future<BOOST_PP_CAT(R, n)>                                      \
    /**/
#define HPX_WHEN_ALL_FUTURE_ARG(z, n, _)                                      \
        lcos::future<BOOST_PP_CAT(R, n)> BOOST_PP_CAT(f, n)                   \
    /**/
#define HPX_WHEN_ALL_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_TYPE, _)>>
    when_all(BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_TYPE, _)>
            result_type;

        return when_n(N, BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_VAR, _), ec);
    }
    
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_TYPE, _)>
    wait_all(BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_TYPE, _)>
            result_type;

        lcos::future<result_type> f = when_all(
            BOOST_PP_ENUM(N, HPX_WHEN_ALL_FUTURE_VAR, _), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_all", 
                "lcos::when_all didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#undef HPX_WHEN_ALL_FUTURE_VAR
#undef HPX_WHEN_ALL_FUTURE_ARG
#undef HPX_WHEN_ALL_FUTURE_TYPE
#undef N

#endif

