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
        template <typename Action, typename Arg0>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Arg0 && arg0)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority),
                std::forward<Arg0>( arg0 ));
            return false; 
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Arg0 && arg0)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ));
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 && arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 )));
            return true; 
        }
        
        template <typename Action, typename Arg0>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::move(arg0)));
            return true; 
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Arg0 && arg0)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ));
        }
    }}
    
    template <typename Action, typename Arg0>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Arg0 && arg0)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), gid,
            priority, std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Arg0>
    inline bool
    apply (naming::id_type const& gid, Arg0 && arg0)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Arg0 && arg0)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c)),
                std::forward<Arg0>( arg0 ));
            return false; 
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_r (naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Arg0 && arg0)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ));
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            Arg0 && arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            actions::continuation_type cont(c);
            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 )));
            return true; 
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, Arg0 && arg0)
        {
            return apply_l_p<Action>(c, target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ));
        }
    }}
    
    template <typename Action, typename Arg0>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), c, gid,
            priority, std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Arg0>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        Arg0 && arg0)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        typename Arg0>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid,
        Arg0 && arg0)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Arg0 && arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Arg0>( arg0 ));
        }
        template <typename Action, typename Arg0>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Arg0 && arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ));
        }
    }}
    template <typename Action, typename Arg0>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Arg0>( arg0 ));
    }
    template <typename Action, typename Arg0>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;
        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )));
            return true; 
        }
        
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 arg0 , Arg1 arg1)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::move(arg0) , std::move(arg1)));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Arg0 && arg0 , Arg1 && arg1)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply (naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_r (naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            actions::continuation_type cont(c);
            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, Arg0 && arg0 , Arg1 && arg1)
        {
            return apply_l_p<Action>(c, target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), c, gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        typename Arg0 , typename Arg1>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        template <typename Action, typename Arg0 , typename Arg1>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
    }}
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Action, typename Arg0 , typename Arg1>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;
        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )));
            return true; 
        }
        
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 arg0 , Arg1 arg1 , Arg2 arg2)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::move(arg0) , std::move(arg1) , std::move(arg2)));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply (naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_r (naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            actions::continuation_type cont(c);
            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            return apply_l_p<Action>(c, target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), c, gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
    }}
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;
        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )));
            return true; 
        }
        
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 arg0 , Arg1 arg1 , Arg2 arg2 , Arg3 arg3)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::move(arg0) , std::move(arg1) , std::move(arg2) , std::move(arg3)));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply (naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_r (naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            actions::continuation_type cont(c);
            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            return apply_l_p<Action>(c, target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), c, gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
    }}
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;
        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
}
namespace hpx
{
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_p(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r (naming::address&& addr, naming::id_type const& gid,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_r_p<Action>(std::move(addr), gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )));
            return true; 
        }
        
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, Arg0 arg0 , Arg1 arg1 , Arg2 arg2 , Arg3 arg3 , Arg4 arg4)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(std::move(arg0) , std::move(arg1) , std::move(arg2) , std::move(arg3) , std::move(arg4)));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_l_p<Action>(target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply (naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r_p(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            
            
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c)),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            return false; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_r (naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_r_p<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;
            HPX_ASSERT(traits::component_type_is_compatible<
                typename action_type::component_type>::call(addr));
            actions::continuation_type cont(c);
            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )));
            return true; 
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            return apply_l_p<Action>(c, target, addr,
                actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }
        
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, std::move(addr),
                priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        
        return applier::detail::apply_r_p<Action>(std::move(addr), c, gid,
            priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > ,
        naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_c_p(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        inline bool
        apply_c (naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;
            return apply_r_p<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
    }}
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;
        return apply_p<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > , naming::id_type const& contgid, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;
        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
}
