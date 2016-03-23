//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/wait_each.hpp

#if !defined(HPX_LCOS_WAIT_EACH_JUN_16_2014_0428PM)
#define HPX_LCOS_WAIT_EACH_JUN_16_2014_0428PM

#if defined(DOXYGEN)
namespace hpx
{
    /// The function \a wait_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    /// Additionally, the supplied function is called for each of the passed
    /// futures as soon as the future has become ready. \a wait_each returns
    /// after all futures have been become ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  A vector holding an arbitrary amount of \a future or
    ///                 \a shared_future objects for which \a wait_each should
    ///                 wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    template <typename F, typename Future>
    void wait_each(F&& f, std::vector<Future>&& futures);

    /// The function \a wait_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    /// Additionally, the supplied function is called for each of the passed
    /// futures as soon as the future has become ready. \a wait_each returns
    /// after all futures have been become ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    /// \param end      The iterator pointing to the last element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    template <typename F, typename Iterator>
    void wait_each(F&& f, Iterator begin, Iterator end);

    /// The function \a wait_each is a operator allowing to join on the results
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    /// Additionally, the supplied function is called for each of the passed
    /// futures as soon as the future has become ready. \a wait_each returns
    /// after all futures have been become ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param futures  An arbitrary number of \a future or \a shared_future
    ///                 objects, possibly holding different types for which
    ///                 \a wait_each should wait.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    template <typename F, typename ...T>
    void wait_each(F&& f, T&&... futures);

    /// The function \a wait_each is a operator allowing to join on the result
    /// of all given futures. It AND-composes all future objects given and
    /// returns after they finished executing.
    /// Additionally, the supplied function is called for each of the passed
    /// futures as soon as the future has become ready.
    ///
    /// \param f        The function which will be called for each of the
    ///                 input futures once the future has become ready.
    /// \param begin    The iterator pointing to the first element of a
    ///                 sequence of \a future or \a shared_future objects for
    ///                 which \a wait_each_n should wait.
    /// \param count    The number of elements in the sequence starting at
    ///                 \a first.
    ///
    /// \note This function consumes the futures as they are passed on to the
    ///       supplied function.
    ///
    template <typename F, typename Iterator>
    void wait_each_n(F&& f, Iterator begin, std::size_t count);
}

#else // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/lcos/when_each.hpp>
#include <hpx/util/detail/pack.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    template <typename F, typename Future>
    void wait_each(F&& f, std::vector<Future>& lazy_values)
    {
        lcos::when_each(std::forward<F>(f), lazy_values).wait();
    }

    template <typename F, typename Future>
    void wait_each(F&& f, std::vector<Future> && lazy_values)
    {
        lcos::when_each(std::forward<F>(f), lazy_values).wait();
    }

    template <typename F, typename Iterator>
    void
    wait_each(F&& f, Iterator begin, Iterator end)
    {
        lcos::when_each(std::forward<F>(f), begin, end).wait();
    }

    template <typename F, typename Iterator>
    void
    wait_each_n(F&& f, Iterator begin, std::size_t count)
    {
        when_each_n(std::forward<F>(f), begin, count).wait();
    }

    template <typename F>
    void wait_each(F&& f)
    {
        lcos::when_each(std::forward<F>(f)).wait();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename boost::disable_if<
        boost::mpl::or_<
            traits::is_future<typename util::decay<F>::type>,
            util::detail::any_of<boost::mpl::not_<traits::is_future<Ts> >...>
        >
    >::type
    wait_each(F&& f, Ts&&... ts)
    {
        lcos::when_each(std::forward<F>(f), std::forward<Ts>(ts)...).wait();
    }
}}

namespace hpx
{
    using lcos::wait_each;
    using lcos::wait_each_n;
}

#endif // DOXYGEN
#endif

