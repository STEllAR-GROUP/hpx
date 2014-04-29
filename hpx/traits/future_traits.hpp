//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_FUTURE_TRAITS_APR_29_2014_0925AM)
#define HPX_TRAITS_FUTURE_TRAITS_APR_29_2014_0925AM

#include <hpx/traits.hpp>

namespace hpx { namespace lcos
{
    template <typename R> class future;
    template <typename R> class shared_future;
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct future_traits
    {};

    template <typename Future>
    struct future_traits<Future const>
      : future_traits<Future>
    {};

    template <typename Future>
    struct future_traits<Future&>
      : future_traits<Future>
    {};

    template <typename Future>
    struct future_traits<Future const &>
      : future_traits<Future>
    {};

    template <typename R>
    struct future_traits<lcos::future<R> >
    {
        typedef R type;
    };

    template <typename R>
    struct future_traits<lcos::shared_future<R> >
    {
        typedef R type;
    };
}}

#endif

