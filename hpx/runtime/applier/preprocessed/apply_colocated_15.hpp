// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <typename Action
       >
    bool apply_colocated(
        naming::id_type const& gid
       )
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                   )
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
       >
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
       )
    {
        return apply_colocated<Derived>(
            gid
           );
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ));
    }
}
namespace hpx
{
    
    template <typename Action
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    bool apply_colocated(
        naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue<action_type>(
            service_target, req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    bool apply_colocated(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14)
    {
        return apply_colocated<Derived>(
            gid
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ));
    }
}
