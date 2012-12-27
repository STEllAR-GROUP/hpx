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
                hpx::detail::when_all<T1, F>(
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
                hpx::detail::when_all<void, F>(
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

