//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_any.hpp

#if !defined(HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM)
#define HPX_LCOS_WAIT_ANY_APR_17_2012_1143AM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns after one future of
    /// that list finishes execution.
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
    /// OR-composes all future objects given and returns after one future of
    /// that list finishes execution.
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
    /// OR-composes all future objects given and returns after one future of
    /// that list finishes execution.
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
    void wait_any(error_code& ec, T&&... futures);

    /// The function \a wait_any is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns after one future of
    /// that list finishes execution.
    ///
    /// \param futures  [in] An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_any should wait.
    ///
    /// \note The function \a wait_any returns after at least one future has
    ///       become ready. All input futures are still valid after \a wait_any
    ///       returns.
    ///
    /// \note     None of the futures in the input sequence are invalidated.
    template <typename ...T>
    void wait_any(T&&... futures);

    /// The function \a wait_any_n is a non-deterministic choice operator. It
    /// OR-composes all future objects given and returns after one future of
    /// that list finishes execution.
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

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_some.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/utility/swap.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    void wait_any(std::vector<Future> const& futures, error_code& ec = throws)
    {
        return lcos::wait_some(1, futures, ec);
    }

    template <typename Future>
    void wait_any(std::vector<Future>& lazy_values, error_code& ec = throws)
    {
        return lcos::wait_any(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Future>
    void wait_any(std::vector<Future> && lazy_values, error_code& ec = throws)
    {
        return lcos::wait_any(
            const_cast<std::vector<Future> const&>(lazy_values), ec);
    }

    template <typename Iterator>
    typename util::always_void<
        typename lcos::detail::future_iterator_traits<Iterator>::type
    >::type
    wait_any(Iterator begin, Iterator end, error_code& ec = throws)
    {
        return lcos::wait_some(1, begin, end, ec);
    }

    inline void wait_any(error_code& ec = throws)
    {
        return lcos::wait_some(1, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator>
    Iterator
    wait_any_n(Iterator begin, std::size_t count,
        error_code& ec = throws)
    {
        return wait_some_n(1, begin, count, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    void wait_any(error_code& ec, Ts&&... ts)
    {
        return lcos::wait_some(1, ec, std::forward<Ts>(ts)...);
    }

    template <typename... Ts>
    void wait_any(Ts&&... ts)
    {
        return lcos::wait_some(1, std::forward<Ts>(ts)...);
    }
}}

namespace hpx
{
    using lcos::wait_any;
    using lcos::wait_any_n;
}

#endif // DOXYGEN
#endif
