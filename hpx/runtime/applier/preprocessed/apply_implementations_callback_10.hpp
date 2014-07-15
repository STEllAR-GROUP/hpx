// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ));
        }
        template <typename Action, typename Callback,
            typename Arg0>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Callback,
        typename Arg0>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            return false; 
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
        }
    }}
    
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        if (addr.locality_ == hpx::get_locality()) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
            cb(boost::system::error_code(), 0); 
            return result;
        }
        
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
        }
        template <typename Action, typename Callback,
            typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
        }
    }}
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
    template <typename Action, typename Callback,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7 , typename Arg8 , typename Arg9>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4 , Arg5 && arg5 , Arg6 && arg6 , Arg7 && arg7 , Arg8 && arg8 , Arg9 && arg9)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ) , std::forward<Arg5>( arg5 ) , std::forward<Arg6>( arg6 ) , std::forward<Arg7>( arg7 ) , std::forward<Arg8>( arg8 ) , std::forward<Arg9>( arg9 ));
    }
}
