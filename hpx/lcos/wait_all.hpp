//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_all.hpp

#if defined(DOXYGEN)
namespace hpx { namespace lcos
{
    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns the same list of futures after they finished executing.
    ///
    /// \param first    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    /// \param last     The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    template <typename InputIter>
    void wait_all(InputIter first, InputIter last);

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns the same list of futures after they finished executing.
    ///
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_all should
    ///                 wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    template <typename R>
    void wait_all(std::vector<future<R>>&& futures);

    /// The function \a wait_all is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns the same list of futures after they finished executing.
    ///
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_all should wait.
    ///
    /// \note The function \a wait_all returns after all futures have become
    ///       ready. All input futures are still valid after \a wait_all
    ///       returns.
    template <typename ...T>
    void wait_all(T &&... futures);
}}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM)
#define HPX_LCOS_WAIT_ALL_APR_19_2012_1140AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_some.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    template <typename Future>
    void wait_all(std::vector<Future> const& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(
            lazy_values.size(), lazy_values, ec);
    }

    template <typename Future>
    void wait_all(std::vector<Future>& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_all(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Future>
    void wait_all(std::vector<Future> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_all(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type wait_all(Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        typedef
            typename lcos::detail::future_iterator_traits<Iterator>::type
            future_type;
        typedef
            typename lcos::detail::shared_state_ptr_for<future_type>::type
            shared_state_ptr;
        typedef std::vector<shared_state_ptr> result_type;

        result_type lazy_values_;
        std::transform(begin, end, std::back_inserter(lazy_values_),
            detail::wait_get_shared_state<future_type>());

        boost::shared_ptr<detail::wait_some<result_type> > f =
            boost::make_shared<detail::wait_some<result_type> >(
                std::move(lazy_values_), lazy_values_.size());

        return (*f.get())();
    }

    template <typename Iterator>
    Iterator wait_all_n(Iterator begin, std::size_t count,
        error_code& ec = throws)
    {
        return wait_some_n(count, begin, count, ec);
    }

    inline void wait_all(error_code& ec = throws)
    {
        return lcos::wait_some(0, ec);
    }
}}

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

namespace hpx
{
    using lcos::wait_all;
    using lcos::wait_all_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    void wait_all(HPX_ENUM_FWD_ARGS(N, T, f), error_code& ec = throws)
    {
        return lcos::wait_some(N, HPX_ENUM_FORWARD_ARGS(N, T, f), ec);
    }
}}

#undef N

#endif

#endif
