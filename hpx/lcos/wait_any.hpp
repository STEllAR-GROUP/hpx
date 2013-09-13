//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WHEN_ANY_APR_17_2012_1143AM

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

#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/tuple.hpp>

#include <boost/atomic.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    /// The function \a when_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned future holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.
    
    template <typename R>
    lcos::future<std::vector<lcos::future<R> > >
    when_any(BOOST_RV_REF(HPX_UTIL_STRIP((
        std::vector<lcos::future<R> >))) lazy_values,
        error_code& ec = throws)
    {
        typedef std::vector<lcos::future<R> > return_type;

        if (lazy_values.empty())
            return lcos::make_ready_future(return_type());

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

    inline lcos::future<HPX_STD_TUPLE<>>
    when_any(error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<> result_type;

        return lcos::make_ready_future(result_type());
    }

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects stored in the given vector and returns
    /// a new future object representing the first future from that list which
    /// finishes execution.
    ///
    /// \return   The returned tuple holds a pair of values, the first value
    ///           is the index of the future which returned first and the second
    ///           value represents the actual future which returned first.

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

#define HPX_WHEN_ANY_FUTURE_TYPE(z, n, _)                                     \
        lcos::future<BOOST_PP_CAT(R, n)>                                      \
    /**/
#define HPX_WHEN_ANY_FUTURE_ARG(z, n, _)                                      \
        lcos::future<BOOST_PP_CAT(R, n)> BOOST_PP_CAT(f, n)                   \
    /**/
#define HPX_WHEN_ANY_FUTURE_VAR(z, n, _) BOOST_PP_CAT(f, n)                   \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    lcos::future<HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)>>
    when_any(BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)>
            result_type;

        return when_n(1, BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _), ec);
    }
    
    template <BOOST_PP_ENUM_PARAMS(N, typename R)>
    HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)>
    wait_any(BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_ARG, _),
        error_code& ec = throws)
    {
        typedef HPX_STD_TUPLE<BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_TYPE, _)>
            result_type;

        lcos::future<result_type> f = when_any(
            BOOST_PP_ENUM(N, HPX_WHEN_ANY_FUTURE_VAR, _), ec);
        if (!f.valid()) {
            HPX_THROWS_IF(ec, uninitialized_value, "lcos::wait_any", 
                "lcos::when_any didn't return a valid future");
            return result_type();
        }

        return f.get(ec);
    }
}

#undef HPX_WHEN_ANY_FUTURE_VAR
#undef HPX_WHEN_ANY_FUTURE_ARG
#undef HPX_WHEN_ANY_FUTURE_TYPE
#undef N

#endif

