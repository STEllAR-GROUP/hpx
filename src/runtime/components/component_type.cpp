//  Copyright (c) 2007-2012 Hartmut Kaiser
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
            "component_runtime_support",                        /*  0 */
            "component_memory",                                 /*  1 */
            "component_memory_block",                           /*  2 */
            "component_base_lco",                               /*  3 */
            "component_base_lco_with_value",                    /*  4 */
            "component_barrier",                                /*  5 */
            "component_promise",                                /*  6 */

            "component_agas_locality_namespace",                /*  7 */
            "component_agas_primary_namespace",                 /*  8 */
            "component_agas_component_namespace",               /*  9 */
            "component_agas_symbol_namespace",                  /* 10 */

#if defined(HPX_HAVE_SODIUM)
            "signed_certificate_promise",                       /* 11 */
            "component_root_certificate_authority",             /* 12 */
            "component_subordinate_certificate_authority",      /* 13 */
#endif
        };
    }

    // Return the string representation for a given component type id
    std::string const get_component_type_name(boost::int32_t type)
    {
        std::string result;

        if (type == component_invalid)
            result = "component_invalid";
        else if ((type < component_last) && (get_derived_type(type) == 0))
            result = components::detail::names[type];
        else if (get_derived_type(type) < component_last && (get_derived_type(type) != 0))
            result = components::detail::names[get_derived_type(type)];
        else
            result = "component";

        if (type == get_base_type(type) || component_invalid == type)
            result += "[" + boost::lexical_cast<std::string>(type) + "]";
        else {
            result += "[" +
                boost::lexical_cast<std::string>
                  (static_cast<int>(get_derived_type(type))) +
                "(" + boost::lexical_cast<std::string>
                    (static_cast<int>(get_base_type(type))) + ")"
                "]";
        }
        return result;
    }
}}

