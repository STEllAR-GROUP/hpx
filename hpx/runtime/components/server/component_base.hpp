//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_BASE_HPP
#define HPX_RUNTIME_COMPONENTS_SERVER_COMPONENT_BASE_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/components/server/simple_component_base.hpp>

namespace hpx { namespace components {
    template <typename Component>
    class component_base
      : public simple_component_base<Component>
    {
    };
}}

#endif
