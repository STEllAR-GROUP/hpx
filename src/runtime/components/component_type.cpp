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
            "gid_promise",                                      /*  7 */
            "vector_gid_romise",                                /*  8 */
            "id_promise",                                       /*  9 */
            "id_gid_promise",                                   /* 10 */
            "vector_id_promise",                                /* 11 */
            "id_vector_gid_vector_promise",                     /* 12 */
            "void_promise",                                     /* 13 */
            "float_promise",                                    /* 14 */
            "double_promise",                                   /* 15 */
            "int8_t_promise",                                   /* 16 */
            "uint8_t_promise",                                  /* 17 */
            "int16_t_promise",                                  /* 18 */
            "uint16_t_promise",                                 /* 19 */
            "int32_t_promise",                                  /* 20 */
            "uint32_t_promise",                                 /* 21 */
            "int64_t_promise",                                  /* 22 */
            "uint64_t_promise",                                 /* 23 */
            "string_promise",                                   /* 24 */
            "bool_promise",                                     /* 25 */
            "section_promise",                                  /* 26 */
            "counter_info_promise",                             /* 27 */
            "counter_value_promise",                            /* 28 */
            "agas_response_promise",                            /* 29 */
            "agas_response_vector_promise",                     /* 30 */
            "id_type_response_promise",                         /* 31 */
            "bool_response_promise",                            /* 32 */
            "uint32_t_response_promise",                        /* 33 */
            "uint32_t_vector_response_promise",                 /* 34 */
            "locality_vector_response_promise",                 /* 35 */
            "memory_data_promise",                              /* 36 */
            "factory_locality_promise",                         /* 37 */

            "component_agas_locality_namespace",                /* 38 */
            "component_agas_primary_namespace",                 /* 39 */
            "component_agas_component_namespace",               /* 40 */
            "component_agas_symbol_namespace",                  /* 41 */

#if defined(HPX_HAVE_SODIUM)
            "signed_certificate_promise",                       /* 42 */
            "component_root_certificate_authority",             /* 43 */
            "component_subordinate_certificate_authority",      /* 44 */
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

