////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F)
#define HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/agas/server/primary_namespace.hpp>

namespace hpx { namespace agas { namespace stubs
{

struct primary_namespace 
{
    // {{{ nested types
    typedef server::primary_namespace server_type; 

    typedef server_type::endpoint_type endpoint_type;
    typedef server_type::gva_type gva_type;
    typedef server_type::count_type count_type;
    typedef server_type::offset_type offset_type;
    typedef server_type::prefix_type prefix_type;
    // }}}

    // {{{ bind_locality dispatch
    static lcos::promise<response>
    bind_locality_async(naming::id_type const& gid, endpoint_type const& ep,
                        count_type count)
    {
        typedef server_type::bind_locality_action action_type;
        return lcos::eager_future<action_type, response>(gid, ep, count);
    }

    static response
    bind_locality(naming::id_type const& gid, endpoint_type const& ep,
                  count_type count, error_code& ec = throws)
    { return bind_locality_async(gid, ep, count).get(ec); } 
    // }}}
    
    // {{{ bind_gid dispatch
    static lcos::promise<response>
    bind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                   gva_type const& gva)
    {
        typedef server_type::bind_gid_action action_type;
        return lcos::eager_future<action_type, response>(gid, id, gva);
    }

    static response
    bind_gid(naming::id_type const& gid, naming::gid_type const& id,
             gva_type const& gva, error_code& ec = throws)
    { return bind_gid_async(gid, id, gva).get(ec); } 
    // }}}

    // {{{ page_fault dispatch 
    static lcos::promise<response>
    page_fault_async(naming::id_type const& gid, naming::gid_type const& key)
    {
        typedef server_type::page_fault_action action_type;
        return lcos::eager_future<action_type, response>(gid, key);
    }
    
    static response
    page_fault(naming::id_type const& gid, naming::gid_type const& key,
                error_code& ec = throws)
    { return page_fault_async(gid, key).get(ec); } 
    // }}}

    // {{{ unbind_locality dispatch 
    static lcos::promise<response>
    unbind_locality_async(naming::id_type const& gid, endpoint_type const& ep)
    {
        typedef server_type::unbind_locality_action action_type;
        return lcos::eager_future<action_type, response>(gid, ep);
    }
    
    static response
    unbind_locality(naming::id_type const& gid, endpoint_type const& ep,
                    error_code& ec = throws)
    { return unbind_locality_async(gid, ep).get(ec); } 
    // }}}

    // {{{ unbind_gid dispatch 
    static lcos::promise<response>
    unbind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                     count_type count)
    {
        typedef server_type::unbind_gid_action action_type;
        return lcos::eager_future<action_type, response>(gid, id, count);
    }
    
    static response
    unbind_gid(naming::id_type const& gid, naming::gid_type const& id,
               count_type count, error_code& ec = throws)
    { return unbind_gid_async(gid, id, count).get(ec); } 
    // }}}
    
    // {{{ increment dispatch 
    static lcos::promise<response>
    increment_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef server_type::increment_action action_type;
        return lcos::eager_future<action_type, response>(gid, key, count);
    }
    
    static response
    increment(naming::id_type const& gid, naming::gid_type const& key,
              count_type count, error_code& ec = throws)
    { return increment_async(gid, key, count).get(ec); } 
    // }}}
    
    // {{{ decrement dispatch 
    static lcos::promise<response>
    decrement_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef server_type::decrement_action action_type;
        return lcos::eager_future<action_type, response>
            (gid, key, count);
    }
    
    static response
    decrement(naming::id_type const& gid, naming::gid_type const& key,
              count_type count, error_code& ec = throws)
    { return decrement_async(gid, key, count).get(ec); } 
    // }}}
    
    // {{{ localities dispatch 
    static lcos::promise<response>
    localities_async(naming::id_type const& gid)
    {
        typedef server_type::localities_action action_type;
        return lcos::eager_future<action_type, response>(gid);
    }
    
    static response
    localities(naming::id_type const& gid, error_code& ec = throws)
    { return localities_async(gid).get(ec); } 
    // }}}
};            

}}}

#endif // HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

