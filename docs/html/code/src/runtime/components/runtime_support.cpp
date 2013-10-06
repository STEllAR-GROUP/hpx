//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/runtime_support.hpp>

namespace hpx { namespace components
{
    int runtime_support::get_factory_properties(components::component_type type)
    {
        return this->base_type::get_factory_properties(gid_, type);
    }

    lcos::future<int>
    runtime_support::get_factory_properties_async(components::component_type type)
    {
        return this->base_type::get_factory_properties_async(gid_, type);
    }
}}


