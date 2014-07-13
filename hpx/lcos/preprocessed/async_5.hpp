// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct sync_local_invoke_1
        {
            template <typename Arg0>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Arg0 && arg0)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Arg0>( arg0 ));
                return p.get_future();
            }
        };
        template <typename Action, typename R>
        struct sync_local_invoke_1<Action, lcos::future<R> >
        {
            template <typename Arg0>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Arg0 && arg0)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 )));
            }
        };
    }
    
    template <typename Action, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Arg0 && arg0)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_1<action_type, result_type>::
                call(gid, addr, std::forward<Arg0>( arg0 ));
        }
        lcos::packaged_action<action_type, result_type> p;
        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Arg0>( arg0 ));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Arg0>( arg0 ));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Arg0>( arg0 ));
            }
        }
        
        
        future<result_type> f = p.get_future();
        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;
            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }
        return std::move(f);
    }
    template <typename Action, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Arg0 && arg0)
    {
        return async<Action>(launch::all, gid,
            std::forward<Arg0>( arg0 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, Arg0 && arg0)
    {
        return async<Derived>(policy, gid,
            std::forward<Arg0>( arg0 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, Arg0 && arg0)
    {
        return async<Derived>(launch::all, gid,
            std::forward<Arg0>( arg0 ));
    }
}
namespace hpx
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct sync_local_invoke_2
        {
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Arg0 && arg0 , Arg1 && arg1)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                return p.get_future();
            }
        };
        template <typename Action, typename R>
        struct sync_local_invoke_2<Action, lcos::future<R> >
        {
            template <typename Arg0 , typename Arg1>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Arg0 && arg0 , Arg1 && arg1)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 )));
            }
        };
    }
    
    template <typename Action, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_2<action_type, result_type>::
                call(gid, addr, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
        }
        lcos::packaged_action<action_type, result_type> p;
        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
            }
        }
        
        
        future<result_type> f = p.get_future();
        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;
            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }
        return std::move(f);
    }
    template <typename Action, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        return async<Action>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        return async<Derived>(policy, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1)
    {
        return async<Derived>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ));
    }
}
namespace hpx
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct sync_local_invoke_3
        {
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                return p.get_future();
            }
        };
        template <typename Action, typename R>
        struct sync_local_invoke_3<Action, lcos::future<R> >
        {
            template <typename Arg0 , typename Arg1 , typename Arg2>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 )));
            }
        };
    }
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_3<action_type, result_type>::
                call(gid, addr, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
        }
        lcos::packaged_action<action_type, result_type> p;
        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
            }
        }
        
        
        future<result_type> f = p.get_future();
        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;
            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }
        return std::move(f);
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return async<Action>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return async<Derived>(policy, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2)
    {
        return async<Derived>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ));
    }
}
namespace hpx
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct sync_local_invoke_4
        {
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                return p.get_future();
            }
        };
        template <typename Action, typename R>
        struct sync_local_invoke_4<Action, lcos::future<R> >
        {
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 )));
            }
        };
    }
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_4<action_type, result_type>::
                call(gid, addr, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
        }
        lcos::packaged_action<action_type, result_type> p;
        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
            }
        }
        
        
        future<result_type> f = p.get_future();
        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;
            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }
        return std::move(f);
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return async<Action>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return async<Derived>(policy, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3)
    {
        return async<Derived>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ));
    }
}
namespace hpx
{
    namespace detail
    {
        template <typename Action, typename Result>
        struct sync_local_invoke_5
        {
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE static lcos::future<Result> call(
                naming::id_type const& gid, naming::address const&,
                Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
            {
                lcos::packaged_action<Action, Result> p;
                p.apply(launch::sync, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                return p.get_future();
            }
        };
        template <typename Action, typename R>
        struct sync_local_invoke_5<Action, lcos::future<R> >
        {
            template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
            BOOST_FORCEINLINE static lcos::future<R> call(
                boost::mpl::true_, naming::id_type const&,
                naming::address const& addr, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
            {
                HPX_ASSERT(traits::component_type_is_compatible<
                    typename Action::component_type>::call(addr));
                return Action::execute_function(addr.address_,
                    util::forward_as_tuple(std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 )));
            }
        };
    }
    
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy, naming::id_type const& gid,
        Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        typedef typename traits::promise_local_result<
            typename action_type::remote_result_type
        >::type result_type;
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr) && policy == launch::sync)
        {
            return detail::sync_local_invoke_5<action_type, result_type>::
                call(gid, addr, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
        }
        lcos::packaged_action<action_type, result_type> p;
        bool target_is_managed = false;
        if (policy == launch::sync || detail::has_async_policy(policy))
        {
            if (addr) {
                p.apply(policy, std::move(addr), gid,
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            }
            else if (gid.get_management_type() == naming::id_type::managed) {
                p.apply(policy,
                    naming::id_type(gid.get_gid(), naming::id_type::unmanaged),
                    std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
                target_is_managed = true;
            }
            else {
                p.apply(policy, gid, std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
            }
        }
        
        
        future<result_type> f = p.get_future();
        if (target_is_managed)
        {
            typedef typename lcos::detail::shared_state_ptr_for<
                future<result_type>
            >::type shared_state_ptr;
            shared_state_ptr const& state = lcos::detail::get_shared_state(f);
            state->set_on_completed(detail::keep_id_alive(gid));
        }
        return std::move(f);
    }
    template <typename Action, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Action>::remote_result_type
        >::type>
    async(naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return async<Action>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(BOOST_SCOPED_ENUM(launch) policy,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const & , naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return async<Derived>(policy, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
    template <typename Component, typename Result,
        typename Arguments, typename Derived, typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    lcos::future<
        typename traits::promise_local_result<
            typename hpx::actions::extract_action<Derived>::remote_result_type
        >::type>
    async(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > const &, naming::id_type const& gid, Arg0 && arg0 , Arg1 && arg1 , Arg2 && arg2 , Arg3 && arg3 , Arg4 && arg4)
    {
        return async<Derived>(launch::all, gid,
            std::forward<Arg0>( arg0 ) , std::forward<Arg1>( arg1 ) , std::forward<Arg2>( arg2 ) , std::forward<Arg3>( arg3 ) , std::forward<Arg4>( arg4 ));
    }
}
