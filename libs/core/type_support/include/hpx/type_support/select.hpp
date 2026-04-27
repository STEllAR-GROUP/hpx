//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

// Contains the metafunction select, which mimics the effect of a chain of
// nested mpl if_'s.
//
// -----------------------------------------------------------------------------
//
// Usage:
//
// typedef typename select<
//                      case1,  type1,
//                      case2,  type2,
//                      ...
//                      true_,  typen
//                  >::type selection;
//
// Here case1, case2, ... are models of MPL::IntegralConstant with value type
// bool, and n <= 12.

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::util {

    HPX_CXX_EXPORT using else_t = std::true_type;

    HPX_CXX_EXPORT struct void_t
    {
        using type = void_t;
    };

    // clang-format off
    HPX_CXX_EXPORT template <
        typename Case1 = std::true_type, typename Type1 = void_t,
        typename Case2 = std::true_type, typename Type2 = void_t,
        typename Case3 = std::true_type, typename Type3 = void_t,
        typename Case4 = std::true_type, typename Type4 = void_t,
        typename Case5 = std::true_type, typename Type5 = void_t,
        typename Case6 = std::true_type, typename Type6 = void_t,
        typename Case7 = std::true_type, typename Type7 = void_t,
        typename Case8 = std::true_type, typename Type8 = void_t,
        typename Case9 = std::true_type, typename Type9 = void_t,
        typename Case10 = std::true_type, typename Type10 = void_t,
        typename Case11 = std::true_type, typename Type11 = void_t,
        typename Case12 = std::true_type, typename Type12 = void_t>
    struct select
    {
        using type =
            lazy_conditional<Case1::value, type_identity<Type1>,
             lazy_conditional<Case2::value, type_identity<Type2>,
              lazy_conditional<Case3::value, type_identity<Type3>,
               lazy_conditional<Case4::value, type_identity<Type4>,
                lazy_conditional<Case5::value, type_identity<Type5>,
                 lazy_conditional<Case6::value, type_identity<Type6>,
                  lazy_conditional<Case7::value, type_identity<Type7>,
                   lazy_conditional<Case8::value, type_identity<Type8>,
                    lazy_conditional<Case9::value, type_identity<Type9>,
                     lazy_conditional<Case10::value, type_identity<Type10>,
                      lazy_conditional<Case11::value, type_identity<Type11>,
                       std::conditional<Case12::value, Type12, void_t>
                      >
                     >
                    >
                   >
                  >
                 >
                >
               >
              >
             >
            >::type;
    };

    HPX_CXX_EXPORT template <
        typename Case1 = std::true_type, typename Type1 = void_t,
        typename Case2 = std::true_type, typename Type2 = void_t,
        typename Case3 = std::true_type, typename Type3 = void_t,
        typename Case4 = std::true_type, typename Type4 = void_t,
        typename Case5 = std::true_type, typename Type5 = void_t,
        typename Case6 = std::true_type, typename Type6 = void_t,
        typename Case7 = std::true_type, typename Type7 = void_t,
        typename Case8 = std::true_type, typename Type8 = void_t,
        typename Case9 = std::true_type, typename Type9 = void_t,
        typename Case10 = std::true_type, typename Type10 = void_t,
        typename Case11 = std::true_type, typename Type11 = void_t,
        typename Case12 = std::true_type, typename Type12 = void_t>
    using select_t = select<
        Case1, Type1, Case2, Type2, Case3, Type3, Case4, Type4,
        Case5, Type5, Case6, Type6, Case7, Type7, Case8, Type8,
        Case9, Type9, Case10, Type10, Case11, Type11, Case12>::type;
    // clang-format on
}    // namespace hpx::util
