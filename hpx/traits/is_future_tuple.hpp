//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_FUTURE_TUPLE_HPP)
#define HPX_TRAITS_IS_FUTURE_TUPLE_HPP

#include <hpx/traits/is_future.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace hpx { namespace traits
{
    template <typename Tuple, typename Enable>
    struct is_future_tuple
      : boost::mpl::false_
    {};

    template <>
    struct is_future_tuple<util::tuple<> >
      : boost::mpl::true_
    {};

#   define HPX_TRAITS_IS_FUTURE_TUPLE_ELEM(Z, N, D)                           \
     && is_future<BOOST_PP_CAT(T, N)>::value                                  \
    /**/
#   define HPX_TRAITS_IS_FUTURE_TUPLE(Z, N, D)                                \
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>                            \
    struct is_future_tuple<util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >          \
      : boost::mpl::bool_<                                                    \
            true BOOST_PP_REPEAT(N, HPX_TRAITS_IS_FUTURE_TUPLE_ELEM, _)       \
        >                                                                     \
    {};                                                                       \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1, BOOST_PP_INC(HPX_TUPLE_LIMIT)
      , HPX_TRAITS_IS_FUTURE_TUPLE, _
    )
    
#   undef HPX_TRAITS_IS_FUTURE_TUPLE_ELEM
#   undef HPX_TRAITS_IS_FUTURE_TUPLE
}}

#endif

