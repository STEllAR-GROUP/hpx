//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_any.hpp

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns the same list of
    /// futures after one future of that list finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_any should wait.
    /// \param last     [in] The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_any should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_any returns after at least one future has
    ///       become ready. All input futures are still valid after \a wait_any
    ///       returns.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter>
    void wait_any(InputIter first, InputIter last, error_code& ec = throws);

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns the same list of
    /// futures after one future of that list finishes execution.
    ///
    /// \param futures  [in] A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_any should
    ///                 wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_any returns after at least one future has
    ///       become ready. All input futures are still valid after \a wait_any
    ///       returns.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename R>
    void wait_any(std::vector<future<R>>& futures, error_code& ec = throws);

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns the same list of
    /// futures after one future of that list finishes execution.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_any should wait.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_any returns after at least one future has
    ///       become ready. All input futures are still valid after \a wait_any
    ///       returns.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename ...T>
    void wait_any(T &&... futures, error_code& ec = throws);

    /// The function \a wait_any_n is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns the same list of
    /// futures after one future of that list finishes execution.
    ///
    /// \param first    [in] The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_any_n should wait.
    /// \param count    [in] The number of elements in the sequence starting at
    ///                 \a first.
    /// \param ec       [in,out] this represents the error status on exit, if
    ///                 this is pre-initialized to \a hpx#throws the function
    ///                 will throw on error instead.
    ///
    /// \note The function \a wait_any_n returns after at least one future has
    ///       become ready. All input futures are still valid after \a wait_any_n
    ///       returns.
    ///
    /// \return         The function \a wait_all_n will return an iterator
    ///                 referring to the first element in the input sequence
    ///                 after the last processed element.
    ///
    /// \note     As long as \a ec is not pre-initialized to \a hpx::throws this
    ///           function doesn't throw but returns the result code using the
    ///           parameter \a ec. Otherwise it throws an instance of
    ///           \a hpx::exception.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename InputIter>
    InputIter wait_any_n(InputIter first, std::size_t count,
        error_code& ec = throws);
}
#else

#if !BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_some.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/enum.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/utility/swap.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_any(std::vector<Future> const& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_some(1, lazy_values, ec);
    }

    template <typename Future>
    void wait_any(std::vector<Future>& lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_any(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Future>
    void wait_any(std::vector<Future> && lazy_values,
        error_code& ec = throws)
    {
        return lcos::wait_any(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type wait_any(Iterator begin, Iterator end,
        error_code& ec = throws)
    {
        return lcos::wait_some(1, begin, end, ec);
    }

    inline void wait_any(error_code& ec = throws)
    {
        return lcos::wait_some(1, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    Iterator wait_any_n(Iterator begin, std::size_t count,
        error_code& ec = throws)
    {
        return wait_some_n(1, begin, count, ec);
    }
}}

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

namespace hpx
{
    using lcos::wait_any;
    using lcos::wait_any_n;
}

#endif

///////////////////////////////////////////////////////////////////////////////
#else // BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>
    void wait_any(HPX_ENUM_FWD_ARGS(N, T, f), error_code& ec = throws)
    {
        return lcos::wait_some(1, HPX_ENUM_FORWARD_ARGS(N, T, f), ec);
    }
}}

#undef N

#endif

#endif
