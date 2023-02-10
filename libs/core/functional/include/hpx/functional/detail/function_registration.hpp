//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2022 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/demangle_helper.hpp>
#include <hpx/preprocessor/stringize.hpp>
#include <hpx/preprocessor/strip_parens.hpp>

#include <type_traits>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename T>
    struct get_function_name_declared : std::false_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename F>
    struct get_function_name_impl
    {
        static char const* call()
#ifdef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        {
            return debug::type_id<F>::typeid_.type_id();
        }
#else
            = delete;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename F>
    [[nodiscard]] char const* get_function_name()
    {
        return get_function_name_impl<VTable, F>::call();
    }
}    // namespace hpx::util::detail

////////////////////////////////////////////////////////////////////////////////
// clang-format off
#define HPX_DECLARE_GET_FUNCTION_NAME(VTable, F, Name)                         \
    namespace hpx::util::detail {                                              \
         template <>                                                           \
         HPX_ALWAYS_EXPORT char const* get_function_name<VTable,               \
             std::decay<HPX_PP_STRIP_PARENS(F)>::type>();                      \
                                                                               \
         template <>                                                           \
         struct get_function_name_declared<VTable,                             \
             std::decay<HPX_PP_STRIP_PARENS(F)>::type> : std::true_type        \
         {                                                                     \
         };                                                                    \
    }                                                                          \
    /**/

#define HPX_DEFINE_GET_FUNCTION_NAME(VTable, F, Name)                          \
    namespace hpx::util::detail {                                              \
        template <>                                                            \
        HPX_ALWAYS_EXPORT char const* get_function_name<VTable,                \
            std::decay<HPX_PP_STRIP_PARENS(F)>::type>()                        \
        {                                                                      \
        /* If you encounter this assert while compiling code, that means that */ \
        /* you have a HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION macro somewhere  */ \
        /* in a source file, but the header in which the function is defined  */ \
        /* misses a HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION_DECLARATION        */ \
             static_assert(                                                    \
                 get_function_name_declared<VTable,                            \
                     std::decay<HPX_PP_STRIP_PARENS(F)>::type>::value,         \
                 "HPX_UTIL_REGISTER_[MOVE_ONLY_]FUNCTION_DECLARATION "         \
                 "missing for " HPX_PP_STRINGIZE(Name));                       \
             return HPX_PP_STRINGIZE(Name);                                    \
        }                                                                      \
    }                                                                          \
    /**/

// clang-format on
