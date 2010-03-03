//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    template<> HPX_ALWAYS_EXPORT component_type
    get_component_type<server::plain_function>()
    { 
        return server::plain_function::get_component_type(); 
    }

    template<> HPX_ALWAYS_EXPORT void
    set_component_type<server::plain_function>(component_type t)
    { 
    }
}}
