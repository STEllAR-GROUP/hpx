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
  : stub_base<server::refcnt_service>
{
    typedef server::refcnt_service::registry_type::key_type key_type;
    typedef server::refcnt_service::registry_type::mapped_type mapped_type; 
    
    typedef server::refcnt_service::decrement_result_type decrement_result_type;

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<key_type>
    increment_async(naming::id_type const& gid, key_type const& key,
                    mapped_type count)
    {
        typedef service::refcnt_service::increment_action action_type;
        return lcos::eager_future<action_type, key_type>(gid, key, count);
    } 
    
    static key_type
    increment(naming::id_type const& gid, key_type const& key,
              mapped_type count)
    {
        return increment_async(gid, key, value).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    static lcos::future_value<decrement_result_type>
    decrement_async(naming::id_type const& gid, key_type const& key,
                    mapped_type count)
    {
        typedef service::refcnt_service::decrement_action action_type;
        return lcos::eager_future<action_type, key_type>(gid, key, count);
    } 
    
    static decrement_result_type
    decrement(naming::id_type const& gid, key_type const& key,
              mapped_type count)
    {
        return decrement_async(gid, key, value).get();
    }
};            

}}}}

#endif // HPX_84B8E09E_9BB8_4E81_A3C5_2424910BDDE6

