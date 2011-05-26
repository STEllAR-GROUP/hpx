////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace agas 
{

// TODO: add error code parameters
template <typename Base, typename Server>
struct primary_namespace_base : Base
{
    // {{{ nested types 
    typedef Base base_type; 
    typedef Server server_type;

    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::offset_type offset_type;
    typedef typename server_type::prefix_type prefix_type;
    typedef typename server_type::prefixes_type prefixes_type;
    typedef typename server_type::binding_type binding_type;
    typedef typename server_type::unbinding_type unbinding_type;
    typedef typename server_type::locality_type locality_type;
    typedef typename server_type::decrement_type decrement_type;
    // }}}

    explicit primary_namespace_base(naming::id_type const& id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind_locality and bind_gid interface 
    lcos::future_value<binding_type>
    bind_async(endpoint_type const& ep, count_type count = 0)
    { return this->base_type::bind_locality_async(this->gid_, ep, count); }

    binding_type bind(endpoint_type const& ep, count_type count = 0)
    { return this->base_type::bind_locality(this->gid_, ep, count); }
    
    lcos::future_value<bool>
    bind_async(naming::gid_type const& gid, gva_type const& gva)
    { return this->base_type::bind_gid_async(this->gid_, gid, gva); }

    bool bind(naming::gid_type const& gid, gva_type const& gva)
    { return this->base_type::bind_gid(this->gid_, gid, gva); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve_endpoint and resolve_gid interface
    lcos::future_value<locality_type>
    resolve_async(endpoint_type const& ep)
    { return this->base_type::resolve_locality_async(this->gid_, ep); }
    
    locality_type resolve(endpoint_type const& ep)
    { return this->base_type::resolve_locality(this->gid_, ep); }

    lcos::future_value<gva_type>
    resolve_async(naming::gid_type const& gid)
    { return this->base_type::resolve_gid_async(this->gid_, gid); }
    
    gva_type resolve(naming::gid_type const& gid)
    { return this->base_type::resolve_gid(this->gid_, gid); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::future_value<unbinding_type>
    unbind_async(naming::gid_type const& gid, count_type count)
    { return this->base_type::unbind_async(this->gid_, gid, count); }
    
    unbinding_type unbind(naming::gid_type const& gid, count_type count)
    { return this->base_type::unbind(this->gid_, gid, count); }
    
    ///////////////////////////////////////////////////////////////////////////
    // increment interface 
    lcos::future_value<count_type>
    increment_async(naming::gid_type const& gid, count_type credits)
    { return this->base_type::increment_async(this->gid_, gid, credits); }
    
    count_type increment(naming::gid_type const& gid, count_type credits)
    { return this->base_type::increment(this->gid_, gid, credits); }
    
    ///////////////////////////////////////////////////////////////////////////
    // decrement interface 
    lcos::future_value<decrement_type>
    decrement_async(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement_async(this->gid_, gid, credits); }
    
    decrement_type 
    decrement(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement(this->gid_, gid, credits); }

    ///////////////////////////////////////////////////////////////////////////
    // localities interface 
    lcos::future_value<prefixes_type> localities_async()
    { return this->base_type::localities_async(this->gid_); }
    
    prefixes_type localities()
    { return this->base_type::localities(this->gid_); }
}; 

}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

