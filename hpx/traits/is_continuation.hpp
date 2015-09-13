//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CONTINUATION_HPP)
#define HPX_TRAITS_IS_CONTINUATION_HPP

#include <hpx/traits.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Continuation, typename Enable = void>
        struct is_continuation_impl
          : boost::mpl::false_
        {};

        template <typename Continuation>
        struct is_continuation_impl<Continuation,
            typename util::always_void<typename Continuation::continuation_tag>::type
        > : boost::mpl::true_
        {};
    }

    template <typename Continuation, typename Enable>
    struct is_continuation
      : detail::is_continuation_impl<typename util::decay<Continuation>::type>
    {};
}}

#endif

