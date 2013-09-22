//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_N_APR_19_2012_0203PM)
#define HPX_LCOS_WAIT_N_APR_19_2012_0203PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/assert.hpp>
#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        template <typename Callback>
        struct set_on_completed_callback_impl
        {
            explicit set_on_completed_callback_impl(Callback const& callback)
              : callback_(callback)
            {}

            template <typename R>
            void operator()(lcos::future<R>& future) const
            {
                using lcos::detail::get_future_data;

                lcos::detail::future_data_base<R>* future_data =
                    get_future_data(future);

                future_data->set_on_completed(Callback(callback_));
            }

            template <typename Sequence>
            void operator()(Sequence& sequence, typename boost::enable_if<
                boost::fusion::traits::is_sequence<Sequence> >::type* = 0) const
            {
                boost::fusion::for_each(sequence, *this);
            }

            template <typename Sequence>
            void operator()(Sequence& sequence, typename boost::disable_if<
                boost::fusion::traits::is_sequence<Sequence> >::type* = 0) const
            {
                std::for_each(sequence.begin(), sequence.end(), *this);
            }

            Callback const& callback_;
        };

        template <typename Sequence>
        struct when_n : boost::enable_shared_from_this<when_n<Sequence> >
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_n)

            void on_future_ready(threads::thread_id_type id)
            {
                if (count_.fetch_add(1) + 1 == needed_count_)
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef Sequence argument_type;
            typedef Sequence result_type;

            when_n(argument_type const& lazy_values, std::size_t n)
              : lazy_values_(lazy_values)
              , count_(0)
              , needed_count_(n)
            {}

            when_n(BOOST_RV_REF(argument_type) lazy_values, std::size_t n)
              : lazy_values_(boost::move(lazy_values))
              , count_(0)
              , needed_count_(n)
            {}

            when_n(BOOST_RV_REF(when_n) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_))
              , needed_count_(rhs.needed_count_)
            {
                rhs.needed_count_ = 0;
            }

            when_n& operator=(BOOST_RV_REF(when_n) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);

                    needed_count_ = rhs.needed_count_;
                    rhs.needed_count_ = 0;
                }
                return *this;
            }

            result_type operator()()
            {
                // set callback functions to executed when future is ready
                threads::thread_id_type id = threads::get_self_id();

                set_on_completed_callback_impl<Callback> set_on_completed_callback_helper(callback);
                set_on_completed_callback_helper(lazy_values_, util::bind(
                        &when_n::on_future_ready, this->shared_from_this(), id));

                // if all of the requested futures are already set, our
                // callback above has already been called, otherwise we suspend
                // ourselves
                if (count_.load() < needed_count_)
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // at least N futures should be ready
                BOOST_ASSERT(count_.load() >= needed_count_);

                return boost::move(lazy_values_);
            }

            result_type lazy_values_;
            boost::atomic<std::size_t> count_;
            std::size_t needed_count_;
        };
    }

    /// The function \a when_n is a operator allowing to join on the result
    /// of n given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new future object representing the same
    /// list of futures after they finished executing.
    ///
    /// \return   The returned future holds the same list of futures as has
    ///           been passed to when_n.

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > >
    when_n(std::size_t n, BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }

        if (n > lazy_values.size())
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter, 
                "hpx::lcos::when_n", 
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }

        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > > //-V659
    when_n(std::size_t n, std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;
        
        result_type lazy_values_(lazy_values);
        return when_n(n, boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<lcos::future<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    > > >
    when_n(std::size_t n, Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef typename lcos::detail::future_iterator_traits<
            Iterator>::traits_type::type value_type;
        typedef std::vector<lcos::future<value_type> > result_type;

        result_type lazy_values_(begin, end);
        return when_n(n, boost::move(lazy_values_), ec);
    }

    inline lcos::future<HPX_STD_TUPLE<> >
    when_n(std::size_t n, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        result_type lazy_values;

        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }

        //if (n > 0)
        //{
            HPX_THROWS_IF(ec, hpx::bad_parameter, 
                "hpx::lcos::when_n", 
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        //}
    }

    /// The function \a wait_n is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects stored in the
    /// given vector and returns a new vector object representing the list of
    /// the first N futures finished executing.
    ///
    /// \return   The returned vector holds the same list of futures as has
    ///           been passed to wait_n.

    template <typename R>
    std::vector<lcos::future<R> >
    wait_n(std::size_t n, BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        lcos::future<result_type> f = when_n(n, lazy_values, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n", 
                "lcos::when_n didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }

    template <typename R>
    std::vector<lcos::future<R> >
    wait_n(std::size_t n, std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        result_type lazy_values_(lazy_values);
        return wait_n(n, boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    std::vector<lcos::future<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    > >
    wait_n(std::size_t n, Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef std::vector<lcos::future<
            typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
        > > result_type;
        
        result_type lazy_values_(begin, end);
        return wait_n(n, boost::move(lazy_values_), ec);
    }

    inline HPX_STD_TUPLE<>
    wait_n(std::size_t n, error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        lcos::future<result_type> f = when_n(n, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n", 
                "lcos::when_n didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
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

#define HPX_WHEN_N_FUTURE_TYPE(z, n, _)                                       \
        lcos::future<BOOST_PP_CAT(R, n)>                                      \
    /**/
#define HPX_WHEN_N_FUTURE_ARG(z, n, _)                                        \
        lcos::future<BOOST_PP_CAT(R, n)> BOOST_PP_CAT(f, n)                   \
    /**/
#define HPX_WHEN_N_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                     \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_TYPE, _)> >
    when_n(std::size_t n, BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_TYPE, _)>
            result_type;

        result_type lazy_values(BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_VAR, _));

        if (n == 0)
        {
            return lcos::make_ready_future(boost::move(lazy_values));
        }

        if (n > N)
        {
            HPX_THROWS_IF(ec, hpx::bad_parameter, 
                "hpx::lcos::when_n", 
                "number of results to wait for is out of bounds");
            return lcos::make_ready_future(result_type());
        }

        boost::shared_ptr<detail::when_n<result_type> > f =
            boost::make_shared<detail::when_n<result_type> >(
                boost::move(lazy_values), n);

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_n<result_type>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_TYPE, _)>
    wait_n(std::size_t n, BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_TYPE, _)>
            result_type;

        lcos::future<result_type> f = when_n(n, 
            BOOST_PP_ENUM(N, HPX_WHEN_N_FUTURE_VAR, _), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_n", 
                "lcos::when_n didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#undef HPX_WHEN_N_FUTURE_VAR
#undef HPX_WHEN_N_FUTURE_ARG
#undef HPX_WHEN_N_FUTURE_TYPE
#undef N

#endif

