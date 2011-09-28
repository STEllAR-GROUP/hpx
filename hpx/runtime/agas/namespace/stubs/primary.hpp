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
#include <hpx/runtime/agas/namespace/server/primary.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol>
struct primary_namespace 
{
    // {{{ nested types
    typedef server::primary_namespace<Database, Protocol> server_type; 

    typedef typename server_type::response_type response_type;
    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::offset_type offset_type;
    typedef typename server_type::prefix_type prefix_type;
    // }}}

    // {{{ bind_locality dispatch
    static lcos::promise<response_type>
    bind_locality_async(naming::id_type const& gid, endpoint_type const& ep,
                        count_type count)
    {
        typedef typename server_type::bind_locality_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, ep, count);
    }

    static response_type
    bind_locality(naming::id_type const& gid, endpoint_type const& ep,
                  count_type count, error_code& ec = throws)
    { return bind_locality_async(gid, ep, count).get(ec); } 
    // }}}
    
    // {{{ bind_gid dispatch
    static lcos::promise<response_type>
    bind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                   gva_type const& gva)
    {
        typedef typename server_type::bind_gid_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, id, gva);
    }

    static response_type
    bind_gid(naming::id_type const& gid, naming::gid_type const& id,
             gva_type const& gva, error_code& ec = throws)
    { return bind_gid_async(gid, id, gva).get(ec); } 
    // }}}

    // {{{ resolve_gid dispatch 
    static lcos::promise<response_type>
    resolve_gid_async(naming::id_type const& gid, naming::gid_type const& key)
    {
        typedef typename server_type::resolve_gid_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key);
    }
    
    static response_type
    resolve_gid(naming::id_type const& gid, naming::gid_type const& key,
                error_code& ec = throws)
    { return resolve_gid_async(gid, key).get(ec); } 
    // }}}

    // {{{ unbind_locality dispatch 
    static lcos::promise<response_type>
    unbind_locality_async(naming::id_type const& gid, endpoint_type const& ep)
    {
        typedef typename server_type::unbind_locality_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, ep);
    }
    
    static response_type
    unbind_locality(naming::id_type const& gid, endpoint_type const& ep,
                    error_code& ec = throws)
    { return unbind_locality_async(gid, ep).get(ec); } 
    // }}}

    // {{{ unbind_gid dispatch 
    static lcos::promise<response_type>
    unbind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                     count_type count)
    {
        typedef typename server_type::unbind_gid_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, id, count);
    }
    
    static response_type
    unbind_gid(naming::id_type const& gid, naming::gid_type const& id,
               count_type count, error_code& ec = throws)
    { return unbind_gid_async(gid, id, count).get(ec); } 
    // }}}
    
    // {{{ increment dispatch 
    static lcos::promise<response_type>
    increment_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::increment_action action_type;
        return lcos::eager_future<action_type, response_type>(gid, key, count);
    }
    
    static response_type
    increment(naming::id_type const& gid, naming::gid_type const& key,
              count_type count, error_code& ec = throws)
    { return increment_async(gid, key, count).get(ec); } 
    // }}}
    
    // {{{ decrement dispatch 
    static lcos::promise<response_type>
    decrement_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::decrement_action action_type;
        return lcos::eager_future<action_type, response_type>
            (gid, key, count);
    }
    
    static response_type
    decrement(naming::id_type const& gid, naming::gid_type const& key,
              count_type count, error_code& ec = throws)
    { return decrement_async(gid, key, count).get(ec); } 
    // }}}
    
    // {{{ localities dispatch 
    static lcos::promise<response_type>
    localities_async(naming::id_type const& gid)
    {
        typedef typename server_type::localities_action action_type;
        return lcos::eager_future<action_type, response_type>(gid);
    }
    
    static response_type
    localities(naming::id_type const& gid, error_code& ec = throws)
    { return localities_async(gid).get(ec); } 
    // }}}
};            

}}}

#endif // HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

