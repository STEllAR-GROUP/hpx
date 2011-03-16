////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_84B8E09E_9BB8_4E81_A3C5_2424910BDDE6)
#define HPX_84B8E09E_9BB8_4E81_A3C5_2424910BDDE6

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/server/refcnt_service.hpp>

namespace hpx { namespace components { namespace agas { namespace stubs
{

struct refcnt_service
  : components::stubs::stub_base<server::refcnt_service>
{
    typedef server::refcnt_service::registry_type::key_type key_type;
    typedef server::refcnt_service::registry_type::mapped_type mapped_type; 
    
    typedef server::refcnt_service::decrement_result_type decrement_result_type;

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<mapped_type>
    increment_async(naming::id_type const& gid, key_type const& key,
                    mapped_type count)
    {
        typedef server::refcnt_service::increment_action action_type;
        return lcos::eager_future<action_type, mapped_type>(gid, key, count);
    } 
    
    static mapped_type
    increment(naming::id_type const& gid, key_type const& key,
              mapped_type count)
    {
        return increment_async(gid, key, count).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<decrement_result_type>
    decrement_async(naming::id_type const& gid, key_type const& key,
                    mapped_type count)
    {
        typedef server::refcnt_service::decrement_action action_type;
        return lcos::eager_future<action_type, decrement_result_type>
            (gid, key, count);
    } 
    
    static decrement_result_type
    decrement(naming::id_type const& gid, key_type const& key,
              mapped_type count)
    {
        return decrement_async(gid, key, count).get();
    }
};            

}}}}

#endif // HPX_84B8E09E_9BB8_4E81_A3C5_2424910BDDE6

