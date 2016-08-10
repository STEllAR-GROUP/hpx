//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_FUTURE_APR_20_2012_0536PM)
#define HPX_TRAITS_IS_FUTURE_APR_20_2012_0536PM

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace lcos
{
    template <typename R> class future;
    template <typename R> class shared_future;
}}

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Future, typename Enable = void>
        struct is_unique_future
          : std::false_type
        {};

        template <typename R>
        struct is_unique_future<lcos::future<R> >
          : std::true_type
        {};

        template <typename Future, typename Enable = void>
        struct is_future_customization_point
          : std::false_type
        {};
    }

    template <typename Future>
    struct is_future
      : detail::is_future_customization_point<Future>
    {};

    template <typename R>
    struct is_future<lcos::future<R> >
      : std::true_type
    {};

    template <typename R>
    struct is_future<lcos::shared_future<R> >
      : std::true_type
    {};
}}

#endif

