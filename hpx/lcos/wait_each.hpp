//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_EACH_JUN_16_2014_0428PM)
#define HPX_LCOS_WAIT_EACH_JUN_16_2014_0428PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/when_each.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    /// The function \a wait_each is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns the same list of futures after they finished executing.
    ///
    /// \a wait_each returns after all futures have been triggered.
    ///
    /// \note There are three variations of wait_each. The first takes a pair
    ///       of InputIterators. The second takes an std::vector of future<R>.
    ///       The third takes any arbitrary number of future<R>, where R need
    ///       not be the same type.
    ///
    /// \return   Returns either nothing or the iterator pointing to the first
    ///           element after the last one.
    ///

    template <typename Future, typename F>
    void wait_each(std::vector<Future>& lazy_values, F && func)
    {
        lcos::when_each(lazy_values, std::forward<F>(func)).wait();
    }

    template <typename Future, typename F>
    void wait_each(std::vector<Future> && lazy_values, F && f)
    {
        lcos::when_each(lazy_values, std::forward<F>(f)).wait();
    }

    template <typename Iterator, typename F>
    void wait_each(Iterator begin, Iterator end, F && f)
    {
        lcos::when_each(begin, end, std::forward<F>(f)).wait();
    }

    template <typename Iterator, typename F>
    void wait_each_n(Iterator begin, std::size_t count, F && f)
    {
        when_each_n(begin, count, std::forward<F>(f)).wait();
    }

    template <typename F>
    inline void wait_each(F && f)
    {
        lcos::when_each(std::forward<F>(f)).wait();
    }
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/lcos/preprocessed/wait_each.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/wait_each_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_WAIT_ARGUMENT_LIMIT, <hpx/lcos/wait_each.hpp>))               \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

namespace hpx
{
    using lcos::wait_each;
    using lcos::wait_each_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T), typename F>
    typename boost::disable_if<
        boost::mpl::or_<
            boost::mpl::not_<traits::is_future<T0> >,
            traits::is_future<F>
        >
    >::type
    wait_each(HPX_ENUM_FWD_ARGS(N, T, f), F && func)
    {
        lcos::when_each(HPX_ENUM_FORWARD_ARGS(N, T, f),
            std::forward<F>(func)).wait();
    }
}}

#undef N

#endif

