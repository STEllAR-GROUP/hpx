//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2025 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/preprocessor.hpp>

#include <type_traits>

////////////////////////////////////////////////////////////////////////////////
// From hpx/functional/function.hpp
#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, F, Name)                   \
    HPX_DECLARE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)               \
    /**/

#define HPX_UTIL_REGISTER_FUNCTION(Sig, F, Name)                               \
    HPX_DEFINE_GET_FUNCTION_NAME(function_vtable<Sig>, F, Name)                \
    /**/

////////////////////////////////////////////////////////////////////////////////
// From hpx/functional/move_only_function.hpp
#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(Sig, F, Name)            \
    HPX_DECLARE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, F, Name)        \
    /**/

#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION(Sig, F, Name)                        \
    HPX_DEFINE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, F, Name)         \
    /**/

////////////////////////////////////////////////////////////////////////////////
// From hpx/functional/detail/function_registration.hpp
#define HPX_DECLARE_GET_FUNCTION_NAME(VTable, F, Name)                         \
    namespace hpx::util::detail {                                              \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_function_name<VTable, std::decay_t<HPX_PP_STRIP_PARENS(F)>>();     \
                                                                               \
        template <>                                                            \
        struct get_function_name_declared<VTable,                              \
            std::decay_t<HPX_PP_STRIP_PARENS(F)>> : std::true_type             \
        {                                                                      \
        };                                                                     \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_FUNCTION_NAME(VTable, F, Name)                          \
    namespace hpx::util::detail {                                              \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const*                                          \
        get_function_name<VTable, std::decay<HPX_PP_STRIP_PARENS(F)>::type>()  \
        {                                                                      \
            /* If you encounter this assert while compiling code, that      */ \
            /* means that you have a HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION */ \
            /* macro somewhere in a source file, but the header in which    */ \
            /* the function is defined misses a                             */ \
            /* HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION_DECLARATION           */ \
            static_assert(get_function_name_declared<VTable,                   \
                              std::decay_t<HPX_PP_STRIP_PARENS(F)>>::value,    \
                "HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION_DECLARATION "          \
                "missing for " HPX_PP_STRINGIZE(Name));                        \
            return HPX_PP_STRINGIZE(Name);                                     \
        }                                                                      \
    }                                                                          \
    /**/
