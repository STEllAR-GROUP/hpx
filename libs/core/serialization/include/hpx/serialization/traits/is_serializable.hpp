//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/traits/brace_initializable_traits.hpp>

#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class has_serialize_adl
    {
        template <typename T1>
        static std::false_type test(...);

        // clang-format off
        template <typename T1,
            typename = decltype(serialize(
                std::declval<hpx::serialization::output_archive&>(),
                std::declval<std::remove_const_t<T1>&>(), 0u))>
        static std::true_type test(int);
        // clang-format on

    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template <typename T>
    inline constexpr bool has_serialize_adl_v = has_serialize_adl<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class has_struct_serialization
    {
        template <typename T1>
        static std::false_type test(...);

        template <typename T1,
            typename = decltype(serialize_struct(
                std::declval<hpx::serialization::output_archive&>(),
                std::declval<std::remove_const_t<T1>&>(), 0u,
                hpx::traits::detail::arity<T1>()))>
        static std::true_type test(int);

    public:
        static constexpr bool value = decltype(test<T>(0))::value;
    };

    template <typename T>
    inline constexpr bool has_struct_serialization_v =
        has_struct_serialization<T>::value;

}    // namespace hpx::traits
