///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_DETAIL_GET_PROXY_TYPE_HPP
#define HPX_COMPUTE_DETAIL_GET_PROXY_TYPE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace compute { namespace detail
{
    template <typename T, typename Enable = void>
    struct get_proxy_type_impl
    {
        typedef T type;
    };

    template <typename T>
    struct get_proxy_type_impl<T,
        typename hpx::util::always_void<
            typename hpx::util::decay<T>::type::proxy_type>::type>
    {
        typedef
            typename hpx::util::decay<T>::type::proxy_type
            proxy_type;
    };

    template <typename T, typename Enable = void>
    struct get_proxy_type
      : get_proxy_type_impl<T>
    {};

//     template <typename T>
//     struct get_proxy_type<T,
//         typename std::enable_if<hpx::traits::is_iterator<T>::value>::type>
//       : get_proxy_type<
//             typename std::iterator_traits<T>::value_type>
//     {};

}}}

#endif
