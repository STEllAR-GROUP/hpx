////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/traits.hpp>
#include <hpx/runtime/agas/namespace/stubs/primary.hpp>

namespace hpx { namespace agas 
{

// TODO: add error code parameters
template <typename Database, typename Protocol>
struct primary_namespace
  : components::client_base<
      primary_namespace<Database, Protocol>,
      stubs::primary_namespace<Database, Protocol>
    >
{
    // {{{ nested types 
    typedef components::client_base<
        primary_namespace<Database, Protocol>,
        stubs::primary_namespace<Database, Protocol>
    > base_type;

    typedef server::primary_namespace<Database, Protocol> server_type;

    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::offset_type offset_type;
    typedef typename server_type::range_type range_type;
    typedef typename server_type::locality_type locality_type;
    typedef typename server_type::decrement_result_type decrement_result_type;
    // }}}

    explicit primary_namespace(naming::id_type const& id = naming::invalid_id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind_locality and bind_gid interface 
    lcos::future_value<range_type>
    bind_async(endpoint_type const& ep, count_type count = 1)
    { return this->base_type::bind_locality_async(this->gid_, ep, count); }

    range_type bind(endpoint_type const& ep, count_type count = 1)
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
    lcos::future_value<bool>
    unbind_async(naming::gid_type const& gid, count_type count)
    { return this->base_type::unbind_async(this->gid_, gid, count); }
    
    bool unbind(naming::gid_type const& gid, count_type count)
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
    lcos::future_value<decrement_result_type>
    decrement_async(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement_async(this->gid_, gid, credits); }
    
    decrement_result_type 
    decrement(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement(this->gid_, gid, credits); }
}; 

}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

