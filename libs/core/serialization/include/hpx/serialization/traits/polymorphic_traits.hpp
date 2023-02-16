//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/concepts/has_xxx.hpp>
#include <hpx/preprocessor/strip_parens.hpp>

#include <type_traits>

namespace hpx::traits {

    namespace detail {

        HPX_HAS_XXX_TRAIT_DEF(serialized_with_id)
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(hpx_serialization_get_name)
    }    // namespace detail

    template <typename T>
    struct is_intrusive_polymorphic : detail::has_hpx_serialization_get_name<T>
    {
    };

    template <typename T>
    inline constexpr bool is_intrusive_polymorphic_v =
        is_intrusive_polymorphic<T>::value;

    template <typename T>
    struct is_nonintrusive_polymorphic : std::false_type
    {
    };

    template <typename T>
    inline constexpr bool is_nonintrusive_polymorphic_v =
        is_nonintrusive_polymorphic<T>::value;

    template <typename T>
    struct is_serialized_with_id : detail::has_serialized_with_id<T>
    {
    };

    template <typename T>
    inline constexpr bool is_serialized_with_id_v =
        is_serialized_with_id<T>::value;
}    // namespace hpx::traits

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                             \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        struct is_nonintrusive_polymorphic<Class> : std::true_type             \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC_TEMPLATE(TEMPLATE, ARG_LIST)       \
    namespace hpx::traits {                                                    \
        HPX_PP_STRIP_PARENS(TEMPLATE)                                          \
        struct is_nonintrusive_polymorphic<HPX_PP_STRIP_PARENS(ARG_LIST)>      \
          : std::true_type                                                     \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/

#define HPX_TRAITS_SERIALIZED_WITH_ID(Class)                                   \
    namespace hpx::traits {                                                    \
        template <>                                                            \
        struct is_serialized_with_id<Class> : std::true_type                   \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/

#define HPX_TRAITS_SERIALIZED_WITH_ID_TEMPLATE(TEMPLATE, ARG_LIST)             \
    namespace hpx::traits {                                                    \
        HPX_PP_STRIP_PARENS(TEMPLATE)                                          \
        struct is_serialized_with_id<HPX_PP_STRIP_PARENS(ARG_LIST)>            \
          : std::true_type                                                     \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/
