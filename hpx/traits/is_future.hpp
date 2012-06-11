//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_FUTURE_APR_20_2012_0536PM)
#define HPX_TRAITS_IS_FUTURE_APR_20_2012_0536PM

#include <hpx/traits.hpp>
#include <boost/mpl/bool.hpp>

namespace hpx { namespace lcos
{
    template <typename Result, typename RemoteResult> class future;
}}

namespace hpx { namespace traits
{
    template <typename Future, typename Enable>
    struct is_future
      : boost::mpl::false_
    {};

    template <typename Result, typename RemoteResult>
    struct is_future<lcos::future<Result, RemoteResult> >
      : boost::mpl::true_
    {};
}}

#endif

