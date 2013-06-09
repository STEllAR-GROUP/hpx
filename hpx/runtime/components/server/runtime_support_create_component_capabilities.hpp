//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_CAPABILITIES_JUN_09_2013_0103PM)
#define HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_CAPABILITIES_JUN_09_2013_0103PM

#include <hpx/runtime/get_lva.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/runtime/components/server/preprocessed/runtime_support_create_component_capabilities.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/runtime_support_create_component_capabilities_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (0, HPX_ACTION_ARGUMENT_LIMIT,                                        \
     "hpx/runtime/components/server/runtime_support_create_component_capabilities.hpp")) \
/**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif // HPX_RUNTIME_SUPPORT_CREATE_COMPONENT_CAPABILITIES_JUN_09_2013_0103PM

#else   // !BOOST_PP_IS_ITERATING

#define N BOOST_PP_ITERATION()

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Actions used to create components with constructors of various arities.
    template <typename Component
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct action_capability_provider<
        BOOST_PP_CAT(components::server::create_component_action, N)<
            Component BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)> >
    {
        // return the required capabilities to invoke the given action
        static components::security::capability call(
            naming::address::address_type lva)
        {
            components::server::runtime_support* rts =
                get_lva<components::server::runtime_support>::call(lva);

            components::component_type const type =
                components::get_component_type<
                    typename Component::wrapped_type>();
            return rts->get_factory_capabilities(type);
        }
    };

    template <typename Component
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)>
    struct action_capability_provider<
        BOOST_PP_CAT(components::server::create_component_direct_action, N)<
            Component BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, A)> >
    {
        static components::security::capability call(
            naming::address::address_type lva)
        {
            components::server::runtime_support* rts =
                get_lva<components::server::runtime_support>::call(lva);

            components::component_type const type =
                components::get_component_type<
                    typename Component::wrapped_type>();
            return rts->get_factory_capabilities(type);
        }
    };
}}

#undef N

#endif  // !BOOST_PP_IS_ITERATING

