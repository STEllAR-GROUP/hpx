//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM)
#define HPX_COMPONENT_COMPONENT_TYPE_MAR_26_2008_1058AM

#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    enum component_type
    {
        component_invalid = -1,
        component_runtime_support = 0,  // runtime support (needed to create components, etc.)
        component_memory,               // general memory address
        component_thread,               // a ParalleX thread

        // LCO's
        component_base_lco,         ///< the base of all LCO's not waiting on a value
        component_base_lco_with_value,  ///< base LCO's blocking on a value
        component_future,           ///< a future executing the action and 
                                    ///< allowing to wait for the result

        component_last,
        component_first_dynamic = component_last
    };
    
    namespace detail
    {
        char const* const names[] =
        {
            "component_invalid",
            "component_runtime_support",
            "component_memory",
            "component_thread",
            "component_base_lco",
            "component_base_lco_with_value",
            "component_future",
        };
    }

    ///
    inline std::string const get_component_type_name(int type)
    {
        std::string result;
        if (type >= component_invalid && type < component_last)
            result = components::detail::names[type+1];
        else
            result = "component";
        result += ": " + type;
        return result;
    }

    ///
    inline bool is_valid_component_type(int type)
    {
        return type > component_invalid && type < component_last;
    }

}}

#endif

