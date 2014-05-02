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
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 0
      , bool
    >::type
    apply_colocated_cb(
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
                    util::bind(util::functional::extract_locality(gid), _2)
                   )
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
       >
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 0
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
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
      , typename Arg0>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 1
      , bool
    >::type
    apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0)
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
                    util::bind(util::functional::extract_locality(gid), _2)
                  , std::forward<Arg0>( arg0 ))
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 1
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 2
      , bool
    >::type
    apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1)
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
                    util::bind(util::functional::extract_locality(gid), _2)
                  , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ))
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 2
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 3
      , bool
    >::type
    apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
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
                    util::bind(util::functional::extract_locality(gid), _2)
                  , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ))
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 3
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 4
      , bool
    >::type
    apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
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
                    util::bind(util::functional::extract_locality(gid), _2)
                  , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ))
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 4
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
}
namespace hpx
{
    
    template <
        typename Action
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    typename boost::enable_if_c<
        util::tuple_size<typename Action::arguments_type>::value == 5
      , bool
    >::type
    apply_colocated_cb(
        naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
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
                    util::bind(util::functional::extract_locality(gid), _2)
                  , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ))
                ));
    }
    
    template <
        typename Component
      , typename Result
      , typename Arguments
      , typename Derived
      , typename Callback
      , typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    typename boost::enable_if_c<
        util::tuple_size<Arguments>::value == 5
      , bool
    >::type
    apply_colocated_cb(
        hpx::actions::action<Component, Result, Arguments, Derived> 
      , naming::id_type const& gid
      , Callback && cb
      , Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_colocated_cb<Derived>(
            gid
          , std::forward<Callback>(cb)
          , std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
}
