//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_POLYMORPHIC_TRAITS_HPP
#define HPX_TRAITS_POLYMORPHIC_TRAITS_HPP

#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace traits {

    namespace is_intrusive_detail {

        struct helper
        {
            boost::uint64_t hpx_serialization_get_name() const;
        };

        template <class T>
        struct helper_composed: T, helper {};

        template <boost::uint64_t (helper::*) () const>
        struct member_function_holder {};

        template <class T, class Ambiguous =
            member_function_holder<
                &helper::hpx_serialization_get_name> >
        struct impl: boost::mpl::true_ {};

        template <class T>
        struct impl<T,
            member_function_holder<
                &helper_composed<T>::hpx_serialization_get_name> >
        : boost::mpl::false_ {};

    } // namespace detail

    template <class T, class Enable = void>
    struct is_intrusive_polymorphic: boost::mpl::false_ {};

    template <class T>
    struct is_intrusive_polymorphic<T,
        typename boost::enable_if<boost::is_class<T> >::type>:
            is_intrusive_detail::impl<T> {};

    template <class T>
    struct is_nonintrusive_polymorphic:
        boost::mpl::false_ {};

}}

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                          \
    namespace hpx { namespace traits {                                      \
        template <>                                                         \
        struct is_nonintrusive_polymorphic<Class>:                          \
            boost::mpl::true_ {};                                           \
    }}                                                                      \
/**/

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(TEMPLATE, ARG_LIST)    \
    namespace hpx { namespace traits {                                      \
        HPX_UTIL_STRIP(TEMPLATE)                                            \
        struct is_nonintrusive_polymorphic<HPX_UTIL_STRIP(ARG_LIST)>        \
          : boost::mpl::true_ {};                                           \
    }}                                                                      \
/**/

#endif
