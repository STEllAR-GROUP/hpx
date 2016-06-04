//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_FUTURE_TUPLE_HPP)
#define HPX_TRAITS_IS_FUTURE_TUPLE_HPP

#include <hpx/traits/is_future.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    template <typename Tuple, typename Enable = void>
    struct is_future_tuple
      : boost::mpl::false_
    {};

    template <typename ...Ts>
    struct is_future_tuple<util::tuple<Ts...> >
      : util::detail::all_of<is_future<Ts>...>
    {};
}}

#endif
