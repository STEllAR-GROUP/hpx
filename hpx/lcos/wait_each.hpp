//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
    void wait_each(F&& f, Iterator begin, Iterator end)
    {
        lcos::when_each(std::forward<F>(f), begin, end).wait();
    }

    template <typename F, typename Iterator>
    void wait_each_n(F&& f, Iterator begin, std::size_t count)
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
            util::functional::any_of<boost::mpl::not_<traits::is_future<Ts> >...>
        >
    >::type
    wait_each(F&& f, Ts&&... ts)
    {
//         lcos::when_each(std::forward<F>(f), ts...).wait();
    }
}}

namespace hpx
{
    using lcos::wait_each;
    using lcos::wait_each_n;
}

#endif

