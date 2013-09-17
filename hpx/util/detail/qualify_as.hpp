//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_QUALIFY_AS_HPP
#define HPX_UTIL_DETAIL_QUALIFY_AS_HPP

#include <hpx/util/decay.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/move/move.hpp>

#include <boost/ref.hpp>

#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_cv.hpp>
#include <boost/type_traits/add_volatile.hpp>
#include <boost/type_traits/add_reference.hpp>

#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename T, typename U>
    struct qualify_as_impl
    {
        typedef T type;
    };
    
    template <typename T, typename U>
    struct qualify_as_impl<T, U const>
      : boost::add_const<typename qualify_as_impl<T, U>::type >
    {};

    template <typename T, typename U>
    struct qualify_as_impl<T, U volatile>
      : boost::add_volatile<typename qualify_as_impl<T, U>::type >
    {};

    template <typename T, typename U>
    struct qualify_as_impl<T, U const volatile>
      : boost::add_cv<typename qualify_as_impl<T, U>::type >
    {};

    template <typename T, typename U>
    struct qualify_as_impl<T, U&>
      : boost::add_reference<typename qualify_as_impl<T, U>::type>
    {};

    template <typename T, typename U>
    struct qualify_as_impl<T, BOOST_FWD_REF(U)>
    {
        typedef BOOST_FWD_REF(HPX_UTIL_STRIP((typename qualify_as_impl<T, U>::type))) type;
    };
    
    ///////////////////////////////////////////////////////////////////////////
    /// creates a type `T` with the (cv-ref)qualifiers of `U`
    template <typename T, typename U, typename Enable = void>
    struct qualify_as
      : qualify_as_impl<T, U>
    {};

    template <typename T, typename U>
    struct qualify_as<T, U
      , typename boost::enable_if<
            boost::is_reference_wrapper<typename util::decay<U>::type>>::type>
      : qualify_as_impl<
            T
          , typename boost::unwrap_reference<typename util::decay<U>::type>::type&
        >
    {};
}}}

#endif
