////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_669C0758_041E_48C9_80CA_93FB6DA221FF)
#define HPX_669C0758_041E_48C9_80CA_93FB6DA221FF

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/server/basic_namespace.hpp>

namespace hpx { namespace components { namespace agas { namespace stubs
{

template <typename Tag>
struct basic_namespace : components::stubs::stub_base<Tag>
{
    typedef typename hpx::agas::traits::key_type<Tag>::type key_type;
    typedef typename hpx::agas::traits::mapped_type<Tag>::type mapped_type;
    
    typedef typename hpx::agas::traits::bind_hook<Tag>::result_type
        bind_result_type;
    typedef typename hpx::agas::traits::update_hook<Tag>::result_type
        update_result_type;
    typedef typename hpx::agas::traits::resolve_hook<Tag>::result_type
        resolve_result_type;
    typedef typename hpx::agas::traits::unbind_hook<Tag>::result_type
        unbind_result_type;

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<bind_result_type>
    bind_async(naming::id_type const& gid, key_type const& key,
               mapped_type const& value)
    {
        typedef typename server::basic_namespace<Tag>::bind_action
            action_type;
        return lcos::eager_future<action_type, bind_result_type>
            (gid, key, value);
    }

    static bind_result_type
    bind(naming::id_type const& gid, key_type const& key,
         mapped_type const& value)
    { return bind_async(gid, key, value).get(); } 

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<update_result_type>
    update_async(naming::id_type const& gid, key_type const& key,
                 mapped_type const& value)
    {
        typedef typename server::basic_namespace<Tag>::update_action
            action_type;
        return lcos::eager_future<action_type, update_result_type>
            (gid, key, value);
    }

    static update_result_type
    update(naming::id_type const& gid, key_type const& key,
           mapped_type const& value)
    { return update_async(gid, key, value).get(); } 

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<resolve_result_type>
    resolve_async(naming::id_type const& gid, key_type const& key)
    {
        typedef typename server::basic_namespace<Tag>::resolve_action
            action_type;
        return lcos::eager_future<action_type, resolve_result_type>(gid, key);
    }
    
    static resolve_result_type
    resolve(naming::id_type const& gid, key_type const& key)
    { return resolve_async(gid, key).get(); } 

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<unbind_result_type>
    unbind_async(naming::id_type const& gid, key_type const& key)
    {
        typedef typename server::basic_namespace<Tag>::resolve_action
            action_type;
        return lcos::eager_future<action_type, unbind_result_type>(gid, key);
    }
    
    static unbind_result_type
    unbind(naming::id_type const& gid, key_type const& key)
    { return unbind_async(gid, key).get(); } 
};            

}}}}

#endif // HPX_669C0758_041E_48C9_80CA_93FB6DA221FF

