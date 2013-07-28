// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace traits
{
    
    
    template <typename Component
       >
    struct action_capability_provider<
        components::server::create_component_action0<
            Component > >
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
    template <typename Component
       >
    struct action_capability_provider<
        components::server::create_component_direct_action0<
            Component > >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0>
    struct action_capability_provider<
        components::server::create_component_action1<
            Component , A0> >
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
    template <typename Component
      , typename A0>
    struct action_capability_provider<
        components::server::create_component_direct_action1<
            Component , A0> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1>
    struct action_capability_provider<
        components::server::create_component_action2<
            Component , A0 , A1> >
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
    template <typename Component
      , typename A0 , typename A1>
    struct action_capability_provider<
        components::server::create_component_direct_action2<
            Component , A0 , A1> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct action_capability_provider<
        components::server::create_component_action3<
            Component , A0 , A1 , A2> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2>
    struct action_capability_provider<
        components::server::create_component_direct_action3<
            Component , A0 , A1 , A2> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct action_capability_provider<
        components::server::create_component_action4<
            Component , A0 , A1 , A2 , A3> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3>
    struct action_capability_provider<
        components::server::create_component_direct_action4<
            Component , A0 , A1 , A2 , A3> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct action_capability_provider<
        components::server::create_component_action5<
            Component , A0 , A1 , A2 , A3 , A4> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct action_capability_provider<
        components::server::create_component_direct_action5<
            Component , A0 , A1 , A2 , A3 , A4> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct action_capability_provider<
        components::server::create_component_action6<
            Component , A0 , A1 , A2 , A3 , A4 , A5> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct action_capability_provider<
        components::server::create_component_direct_action6<
            Component , A0 , A1 , A2 , A3 , A4 , A5> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct action_capability_provider<
        components::server::create_component_action7<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct action_capability_provider<
        components::server::create_component_direct_action7<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct action_capability_provider<
        components::server::create_component_action8<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct action_capability_provider<
        components::server::create_component_direct_action8<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct action_capability_provider<
        components::server::create_component_action9<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    struct action_capability_provider<
        components::server::create_component_direct_action9<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct action_capability_provider<
        components::server::create_component_action10<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    struct action_capability_provider<
        components::server::create_component_direct_action10<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct action_capability_provider<
        components::server::create_component_action11<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    struct action_capability_provider<
        components::server::create_component_direct_action11<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct action_capability_provider<
        components::server::create_component_action12<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    struct action_capability_provider<
        components::server::create_component_direct_action12<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct action_capability_provider<
        components::server::create_component_action13<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    struct action_capability_provider<
        components::server::create_component_direct_action13<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    struct action_capability_provider<
        components::server::create_component_action14<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    struct action_capability_provider<
        components::server::create_component_direct_action14<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    struct action_capability_provider<
        components::server::create_component_action15<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    struct action_capability_provider<
        components::server::create_component_direct_action15<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    struct action_capability_provider<
        components::server::create_component_action16<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    struct action_capability_provider<
        components::server::create_component_direct_action16<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    struct action_capability_provider<
        components::server::create_component_action17<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    struct action_capability_provider<
        components::server::create_component_direct_action17<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    struct action_capability_provider<
        components::server::create_component_action18<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    struct action_capability_provider<
        components::server::create_component_direct_action18<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    struct action_capability_provider<
        components::server::create_component_action19<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18>
    struct action_capability_provider<
        components::server::create_component_direct_action19<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18> >
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
namespace hpx { namespace traits
{
    
    
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    struct action_capability_provider<
        components::server::create_component_action20<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19> >
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
    template <typename Component
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19>
    struct action_capability_provider<
        components::server::create_component_direct_action20<
            Component , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19> >
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
