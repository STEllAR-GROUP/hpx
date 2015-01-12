// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    template <
        typename Action
      , typename Callback
       >
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
       )
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                   )
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
       >
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
       )
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
           );
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0>
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1>
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2>
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        
        
        agas::request req(agas::primary_ns_resolve_gid, gid.get_gid());
        naming::id_type service_target(
            agas::stubs::primary_namespace::get_service_instance(gid.get_gid())
          , naming::id_type::unmanaged);
        typedef agas::server::primary_namespace::service_action action_type;
        using util::placeholders::_2;
        return apply_continue_cb<action_type>(
            service_target
          , std::forward<Callback>(cb)
          , req
          , util::functional::apply_continuation(
                util::bind<Action>(
                    util::bind(util::functional::extract_locality(), _2, gid)
                  , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ))
                ));
    }
    
    template <
        typename Component, typename Signature, typename Derived
      , typename Callback
      , typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    bool apply_colocated_cb(
        hpx::actions::basic_action<Component, Signature, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ));
    }
}
