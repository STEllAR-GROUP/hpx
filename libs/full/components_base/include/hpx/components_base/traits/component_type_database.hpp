//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx { namespace components {

    typedef std::int32_t component_type;
}}    // namespace hpx::components

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Enable = void>
    struct component_type_database
    {
        static components::component_type value;

        HPX_ALWAYS_EXPORT static components::component_type get();
        HPX_ALWAYS_EXPORT static void set(components::component_type);
    };

    template <typename Component, typename Enable>
    components::component_type
        component_type_database<Component, Enable>::value =
            components::component_type(-1);    //components::component_invalid;

    template <typename Component, typename Enable>
    struct component_type_database<Component const, Enable>
      : component_type_database<Component, Enable>
    {
    };
}}    // namespace hpx::traits
