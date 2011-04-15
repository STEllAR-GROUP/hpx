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

template <typename Database, typename Protocol>
struct primary_namespace
  : client_base<
      primary_namespace<Database, Protocol>,
      stubs::primary_namespace<Database, Protocol>
    >
{
    // {{{ nested types 
    typedef client_base<
        primary_namespace<Database, Protocol>,
        stubs::primary_namespace<Database, Protocol>
    > base_type;

    typedef server::primary_namespace<Database, Protocol> server_type;

    typedef typename server_type::endpoint_type endpoint_type;
    typedef typename server_type::gva_type gva_type;
    typedef typename server_type::full_gva_type full_gva_type;
    typedef typename server_type::count_type count_type;
    typedef typename server_type::components_type components_type;
    // }}}

    explicit primary_namespace(naming::id_type const& id = naming::invalid_id)
      : base_type(id) {}

    ///////////////////////////////////////////////////////////////////////////
    // bind interface 
    lcos::future_value<partition>
    bind_async(gva_type const& gva, count_type count)
    { return this->base_type::bind_async(this->gid_, gva); }

    partition bind(gva_type const& gva, count_type count)
    { return this->base_type::bind(this->gid_, gva, count); }
    
    ///////////////////////////////////////////////////////////////////////////
    // rebind interface 
    lcos::future_value<partition>
    rebind_async(endpoint_type const& ep, count_type count)
    { return this->base_type::rebind_async(this->gid_, ep, count); }

    partition rebind(endpoint_type const& ep, count_type count)
    { return this->base_type::rebind(this->gid_, ep, count); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve_endpoint interface
    lcos::future_value<partition>
    resolve_endpoint_async(endpoint_type const& ep)
    { return this->base_type::resolve_endpoint_async(this->gid_, ep); }
    
    partition resolve_endpoint(endpoint_type const& ep)
    { return this->base_type::resolve_endpoint(this->gid_, ep); }

    ///////////////////////////////////////////////////////////////////////////
    // resolve_gid interface 
    lcos::future_value<gva_type>
    resolve_gid_async(naming::gid_type const& gid)
    { return this->base_type::resolve_gid_async(this->gid_, gid); }
    
    gva_type resolve_gid(naming::gid_type const& gid)
    { return this->base_type::resolve_gid(this->gid_, gid); }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind interface 
    lcos::future_value<bool>
    unbind_async(endpoint_type const& ep, count_type count)
    { return this->base_type::unbind_async(this->gid_, ep, count); }
    
    bool unbind(endpoint_type const& ep, count_type count)
    { return this->base_type::unbind(this->gid_, ep, count); }
    
    ///////////////////////////////////////////////////////////////////////////
    // increment interface 
    lcos::future_value<count_type>
    increment_async(naming::gid_type const& gid, count_type credits)
    { return this->base_type::increment_async(this->gid_, gid, credits); }
    
    count_type increment(naming::gid_type const& gid, count_type credits)
    { return this->base_type::increment(this->gid_, gid, credits); }
    
    ///////////////////////////////////////////////////////////////////////////
    // decrement interface 
    lcos::future_value<boost::fusion::vector2<count_type, component_type> >
    decrement_async(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement_async(this->gid_, gid, credits); }
    
    boost::fusion::vector2<count_type, component_type>
    decrement(naming::gid_type const& gid, count_type credits)
    { return this->base_type::decrement(this->gid_, gid, credits); }
}; 

}}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

