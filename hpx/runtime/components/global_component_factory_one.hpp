//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1977BE08_FA61_4E9C_8753_86A63838FC75)
#define HPX_1977BE08_FA61_4E9C_8753_86A63838FC75

#include <hpx/runtime/components/global_component_factory.hpp>
#include <hpx/runtime/components/component_factory_one.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    template <typename Component, typename StartFunctions, typename StopFunctions>
    struct global_component_factory_one : component_factory_one<Component>
    {
        typedef component_factory_one<Component> base_type;

        typedef StartFunctions start_functions_type; 
        typedef StopFunctions start_functions_type; 

        struct invoker
        {
            template <typename F>
            void operator() (F f) const
            { f(); } 
        };

        global_component_factory_one(util::section const* global,
                                     util::section const* local)
          : base_type(global, local)
        {
            boost::mpl::for_each<start_functions_type>(invoker());
        }

        ~global_component_factory_one()
        {
            boost::mpl::for_each<stop_functions_type>(invoker());
        }
    };
}}

#define HPX_REGISTER_GLOBAL_COMPONENT_FACTORY_ONE(type, starts, stops, name)  \
        typedef HPX_FUNCTION_LIST(starts) BOOST_PP_CAT(name, _start_list);    \
        typedef HPX_FUNCTION_LIST(stops) BOOST_PP_CAT(name, _stop_list);      \
        typedef hpx::components::global_component_factory_one<type,           \
            BOOST_PP_CAT(name, _start_list), BOOST_PP_CAT(name, _stop_list)   \
        > BOOST_PP_CAT(name, _component_type);                                \
        HPX_REGISTER_COMPONENT_FACTORY(                                       \
            BOOST_PP_CAT(name, _component_type), name);                       \
        HPX_DEF_UNIQUE_COMPONENT_NAME(                                        \
            BOOST_PP_CAT(name, _component_type), name)                        \
        template struct hpx::components::global_component_factory_one<type,   \
            BOOST_PP_CAT(name, _start_list), BOOST_PP_CAT(name, _stop_list)>; \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY(type, name)                   \
    /**/

#endif

