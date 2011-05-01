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
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/server/primary.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol>
struct primary_namespace
  : components::stubs::stub_base<server::primary_namespace<Database, Protocol> >
{
    // {{{ nested types
    typedef server::primary_namespace<Database, Protocol> server_type;

    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::offset_type offset_type;
    typedef typename server_type::range_type range_type;
    typedef typename server_type::decrement_result_type decrement_result_type;
    // }}}

    // {{{ bind_locality dispatch
    static lcos::future_value<range_type>
    bind_locality_async(naming::id_type const& gid, endpoint_type const& ep,
                        count_type count)
    {
        typedef typename server_type::bind_locality_action action_type;
        return lcos::eager_future<action_type, range_type>(gid, ep, count);
    }

    static range_type
    bind_locality(naming::id_type const& gid, endpoint_type const& ep,
                  count_type count)
    { return bind_locality_async(gid, ep, count).get(); } 
    // }}}
    
    // {{{ bind_gid dispatch
    static lcos::future_value<bool>
    bind_gid_async(naming::id_type const& gid, naming::gid_type const& id,
                   gva_type const& gva)
    {
        typedef typename server_type::bind_gid_action action_type;
        return lcos::eager_future<action_type, bool>(gid, id, gva);
    }

    static bool
    bind_gid(naming::id_type const& gid, naming::gid_type const& id,
             gva_type const& gva)
    { return bind_gid_async(gid, id, gva).get(); } 
    // }}}

    // {{{ resolve_locality dispatch
    static lcos::future_value<gva_type>
    resolve_locality_async(naming::id_type const& gid, endpoint_type const& ep)
    {
        typedef typename server_type::resolve_locality_action action_type;
        return lcos::eager_future<action_type, gva_type>(gid, ep);
    }
    
    static gva_type
    resolve_locality(naming::id_type const& gid, endpoint_type const& ep)
    { return resolve_locality_async(gid, ep).get(); } 
    // }}}

    // {{{ resolve_gid dispatch 
    static lcos::future_value<gva_type>
    resolve_gid_async(naming::id_type const& gid, naming::gid_type const& key)
    {
        typedef typename server_type::resolve_gid_action action_type;
        return lcos::eager_future<action_type, gva_type>(gid, key);
    }
    
    static gva_type
    resolve_gid(naming::id_type const& gid, naming::gid_type const& key)
    { return resolve_gid_async(gid, key).get(); } 
    // }}}

    // {{{ unbind dispatch 
    static lcos::future_value<void>
    unbind_async(naming::id_type const& gid, naming::gid_type const& id,
                 count_type count)
    {
        typedef typename server_type::unbind_action action_type;
        return lcos::eager_future<action_type, void>(gid, id, count);
    }
    
    static void unbind(naming::id_type const& gid, naming::gid_type const& id,
                       count_type count)
    { return unbind_async(gid, id, count).get(); } 
    // }}}
    
    // {{{ increment dispatch 
    static lcos::future_value<count_type>
    increment_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::increment_action action_type;
        return lcos::eager_future<action_type, count_type>(gid, key, count);
    }
    
    static count_type
    increment(naming::id_type const& gid, naming::gid_type const& key,
              count_type count)
    { return increment_async(gid, key, count).get(); } 
    // }}}
    
    // {{{ decrement dispatch 
    static lcos::future_value<decrement_result_type>
    decrement_async(naming::id_type const& gid, naming::gid_type const& key,
                    count_type count)
    {
        typedef typename server_type::decrement_action action_type;
        return lcos::eager_future<action_type, decrement_result_type>
            (gid, key, count);
    }
    
    static decrement_result_type
    decrement(naming::id_type const& gid, naming::gid_type const& key,
              count_type count)
    { return decrement_async(gid, key, count).get(); } 
    // }}}
};            

}}}

#endif // HPX_5D993B14_5B65_4231_A84E_90AD1807EB8F

