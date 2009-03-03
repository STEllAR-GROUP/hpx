//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    namespace detail
    {
        // the entries in this array need to be in exactly the same sequence
        // as the values defined in the component_type enumerator
        char const* const names[] =
        {
            "component_invalid",
            "component_runtime_support",
            "component_memory",
            "component_memory_block",
            "component_thread",
            "component_base_lco",
            "component_base_lco_with_value",
            "component_future",
            "component_value_adaptor",
            "component_performance_counter",
        };
    }

    // Return the string representation for a given component type id
    std::string const get_component_type_name(int type)
    {
        std::string result;
        if (type >= component_invalid && type < component_last)
            result = components::detail::names[type+1];
        else
            result = "component";

        if (type == get_base_type(type) || component_invalid == type)
            result += "[" + boost::lexical_cast<std::string>(type) + "]";
        else {
            result += "[" + 
                boost::lexical_cast<std::string>(get_derived_type(type)) + 
                "(" + boost::lexical_cast<std::string>(get_base_type(type)) + ")"
                "]";
        }
        return result;
    }

}}

