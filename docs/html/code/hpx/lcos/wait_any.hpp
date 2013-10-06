//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_n.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/utility/swap.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct when_any_swapped : boost::enable_shared_from_this<when_any_swapped<T> >
        {
        private:
            BOOST_MOVABLE_BUT_NOT_COPYABLE(when_any_swapped)

            enum { index_error = -1 };

            void on_future_ready(std::size_t idx, threads::thread_id_type id)
            {
                std::size_t index_not_initialized =
                    static_cast<std::size_t>(index_error);
                if (index_.compare_exchange_strong(index_not_initialized, idx))
                {
                    // reactivate waiting thread only if it's not us
                    if (id != threads::get_self_id())
                        threads::set_thread_state(id, threads::pending);
                }
            }

        public:
            typedef std::vector<lcos::future<T> > result_type;
            typedef std::vector<lcos::future<T> > argument_type;

            when_any_swapped(argument_type const& lazy_values)
              : lazy_values_(lazy_values)
              , index_(static_cast<std::size_t>(index_error))
            {}

            when_any_swapped(BOOST_RV_REF(argument_type) lazy_values)
              : lazy_values_(boost::move(lazy_values))
              , index_(static_cast<std::size_t>(index_error))
            {}

            when_any_swapped(BOOST_RV_REF(when_any_swapped) rhs)
              : lazy_values_(boost::move(rhs.lazy_values_))
              , index_(rhs.index_.load())
            {
                rhs.needed_count_ = 0;
            }

            when_any_swapped& operator=(BOOST_RV_REF(when_any_swapped) rhs)
            {
                if (this != &rhs) {
                    lazy_values_ = boost::move(rhs.lazy_values_);

                    index_ = rhs.index_;
                    rhs.index_.store(static_cast<std::size_t>(index_error));
                }
                return *this;
            }

            result_type operator()()
            {
                using lcos::detail::get_future_data;

                index_.store(static_cast<std::size_t>(index_error));

                std::size_t size = lazy_values_.size();

                // set callback functions to execute when future is ready
                threads::thread_id_type id = threads::get_self_id();
                for (std::size_t i = 0; i != size; ++i)
                {
                    lcos::detail::future_data_base<T>* current =
                        get_future_data(lazy_values_[i]);

                    current->set_on_completed(util::bind(
                        &when_any_swapped::on_future_ready, this->shared_from_this(), i, id));
                }

                // If one of the futures is already set, our callback above has
                // already been called, otherwise we suspend ourselves
                if (index_.load() == static_cast<std::size_t>(index_error))
                {
                    // wait for any of the futures to return to become ready
                    this_thread::suspend(threads::suspended);
                }

                // that should not happen
                BOOST_ASSERT(index_.load() != static_cast<std::size_t>(index_error));

                boost::swap(lazy_values_[index_], lazy_values_.back());
                return boost::move(lazy_values_);
            }

            std::vector<lcos::future<T> > lazy_values_;
            boost::atomic<std::size_t> index_;
        };
    }

    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns a new future object
    /// representing the same list of futures after one future of that list
    /// finishes execution.
    ///
    /// \note There are three variations of when_any. The first takes a pair
    ///       of InputIterators. The second takes an std::vector of future<R>.
    ///       The third takes any arbitrary number of future<R>, where R need
    ///       not be the same type.
    ///
    /// \return   Returns a future holding the same list of futures as has
    ///           been passed to when_any.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///           - future<tuple<future<R0>, future<R1>, future<R2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > >
    when_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(result_type());

        return when_n(1, lazy_values, ec);
    }

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > > //-V659
    when_any(std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        result_type lazy_values_(lazy_values);
        return when_any(boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<lcos::future<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    > > >
    when_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef typename lcos::detail::future_iterator_traits<
            Iterator>::traits_type::type value_type;
        typedef std::vector<lcos::future<value_type> > result_type;

        result_type lazy_values_(begin, end);
        return when_any(boost::move(lazy_values_), ec);
    }

    inline lcos::future<HPX_STD_TUPLE<> > //-V524
    when_any(error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        return lcos::make_ready_future(result_type());
    }

    /// The function \a when_any_swapped is a non-deterministic choice
    /// operator. It OR-composes all future objects given and returns the same
    /// list of futures after one future of that list finishes execution. The
    /// future object that was first detected as being ready swaps its
    /// position with that of the last element of the result collection, so
    /// that the ready future object may be identified in constant time.
    ///
    /// \note There are two variations of when_any_swapped. The first takes
    ///       a pair of InputIterators. The second takes an std::vector of
    ///       future<R>.
    ///
    /// \return   The same list of futures as has been passed to
    ///           when_any_swapped, where the future object that was first
    ///           detected as being ready has swapped position with the last
    ///           element in the list.

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > >
    when_any_swapped(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(boost::move(lazy_values));

        boost::shared_ptr<detail::when_any_swapped<R> > f =
            boost::make_shared<detail::when_any_swapped<R> >(
                boost::move(lazy_values));

        lcos::local::futures_factory<result_type()> p(
            util::bind(&detail::when_any_swapped<R>::operator(), f));

        p.apply();
        return p.get_future();
    }

    template <typename R>
    lcos::future<std::vector<lcos::future<R> > > //-V659
    when_any_swapped(std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        result_type lazy_values_(lazy_values);
        return when_any_swapped(boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    lcos::future<std::vector<lcos::future<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    > > >
    when_any_swapped(Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef typename lcos::detail::future_iterator_traits<
            Iterator>::traits_type::type value_type;
        typedef std::vector<lcos::future<value_type> > result_type;

        result_type lazy_values_(begin, end);
        return when_any_swapped(boost::move(lazy_values_), ec);
    }

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns the same list of
    /// futures after one future of that list finishes execution.
    ///
    /// \a wait_any returns after one future has been triggered.
    ///
    /// \note There are three variations of wait_any. The first takes a pair
    ///       of InputIterators. The second takes an std::vector of future<R>.
    ///       The third takes any arbitrary number of future<R>, where R need
    ///       not be the same type.
    ///
    /// \return   The same list of futures as has been passed to wait_any.
    ///           - future<vector<future<R>>>: If the input cardinality is
    ///             unknown at compile time and the futures are all of the
    ///             same type.
    ///           - future<tuple<future<R0>, future<R1>, future<R2>...>>: If
    ///             inputs are fixed in number and are of heterogeneous types.
    ///             The inputs can be any arbitrary number of future objects.

    template <typename R>
    std::vector<lcos::future<R> >
    wait_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        lcos::future<result_type> f = when_any(lazy_values, ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }

    template <typename R>
    std::vector<lcos::future<R> >
    wait_any(std::vector<lcos::future<R> > const& lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > result_type;

        result_type lazy_values_(lazy_values);
        return wait_any(boost::move(lazy_values_), ec);
    }

    template <typename Iterator>
    std::vector<
        typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
    >
    wait_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        typedef std::vector<lcos::future<
            typename lcos::detail::future_iterator_traits<Iterator>::traits_type::type
        > > result_type;

        result_type lazy_values_(begin, end);
        return wait_any(boost::move(lazy_values_), ec);
    }

    inline HPX_STD_TUPLE<>
    wait_any(error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        lcos::future<result_type> f = when_any(ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_any.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_any_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_any.hpp>))                \
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

#define HPX_WAIT_ANY_FUTURE_TYPE(z, n, _)                                     \
        lcos::future<BOOST_PP_CAT(R, n)>                                      \
    /**/
#define HPX_WAIT_ANY_FUTURE_ARG(z, n, _)                                      \
        lcos::future<BOOST_PP_CAT(R, n)> BOOST_PP_CAT(f, n)                   \
    /**/
#define HPX_WAIT_ANY_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_TYPE, _)> >
    when_any(BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        return when_n(1, BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_VAR, _), ec);
    }

    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_TYPE, _)>
    wait_any(BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_TYPE, _)>
            result_type;

        lcos::future<result_type> f = when_any(
            BOOST_PP_ENUM(N, HPX_WAIT_ANY_FUTURE_VAR, _), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any",
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#undef HPX_WAIT_ANY_FUTURE_VAR
#undef HPX_WAIT_ANY_FUTURE_ARG
#undef HPX_WAIT_ANY_FUTURE_TYPE
#undef N

#endif

