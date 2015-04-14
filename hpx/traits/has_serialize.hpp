//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_HAS_SERIALIZE_HPP
#define HPX_TRAITS_HAS_SERIALIZE_HPP

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace traits {

    namespace has_serialize_detail {
   
        struct helper
        {
            void serialize(...);
        };
   
        template <class T>
        struct helper_composed: T, helper {};
   
        template <void (helper::*) (...)>
        struct member_function_holder {};
   
        template <class T, class Ambiguous =
            member_function_holder<&helper::serialize> >
        struct impl: boost::mpl::true_ {};
   
        template <class T>
        struct impl<T, 
            member_function_holder<&helper_composed<T>::serialize> >
        : boost::mpl::false_ {};
   
    } // namespace detail
   
    template <class T, class Enable = void>
    struct has_serialize: boost::mpl::false_ {};
   
    template <class T>
    struct has_serialize<T,
        typename boost::enable_if<boost::is_class<T> >::type>:
            has_serialize_detail::impl<T> {};

}}

#endif
