//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_GET_FUNCTION_ANNOTATION_JAN_31_2017_1201PM)
#define HPX_TRAITS_GET_FUNCTION_ANNOTATION_JAN_31_2017_1201PM

#include <hpx/config.hpp>

#include <cstddef>
#include <memory>

namespace hpx { namespace traits
{
    // By default we don't know anything about the function's name
    template <typename F, typename Enable = void>
    struct get_function_annotation
    {
        static char const* call(F const& f) HPX_NOEXCEPT
        {
            return nullptr;
        }
    };
}}

#endif
