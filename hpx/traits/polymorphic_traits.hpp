//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_POLYMORPHIC_TRAITS_HPP
#define HPX_TRAITS_POLYMORPHIC_TRAITS_HPP

#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/traits/has_xxx.hpp>
#include <hpx/util/detail/pp/strip_parens.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    namespace detail
    {
        HPX_HAS_XXX_TRAIT_DEF(serialized_with_id);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(hpx_serialization_get_name);
    } // namespace detail

    template <typename T>
    struct is_intrusive_polymorphic
      : detail::has_hpx_serialization_get_name<T> {};

    template <typename T>
    struct is_nonintrusive_polymorphic
      : std::false_type {};

    template <typename T>
    struct is_serialized_with_id
      : detail::has_serialized_with_id<T> {};
}}

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                            \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct is_nonintrusive_polymorphic<Class>                             \
          : std::true_type {};                                                \
    }}                                                                        \
/**/

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(TEMPLATE, ARG_LIST)      \
    namespace hpx { namespace traits {                                        \
        HPX_PP_STRIP_PARENS(TEMPLATE)                                         \
        struct is_nonintrusive_polymorphic<HPX_PP_STRIP_PARENS(ARG_LIST)>     \
          : std::true_type {};                                                \
    }}                                                                        \
/**/

#define HPX_TRAITS_SERIALIZED_WITH_ID(Class)                                  \
    namespace hpx { namespace traits {                                        \
        template <>                                                           \
        struct is_serialized_with_id<Class>                                   \
          : std::true_type {};                                                \
    }}                                                                        \
/**/

#define HPX_TRAITS_SERIALIZED_WITH_ID_TEMPLATE(TEMPLATE, ARG_LIST)            \
    namespace hpx { namespace traits {                                        \
        HPX_PP_STRIP_PARENS(TEMPLATE)                                         \
        struct is_serialized_with_id<HPX_PP_STRIP_PARENS(ARG_LIST)>           \
          : std::true_type {};                                                \
    }}                                                                        \
/**/

#endif
