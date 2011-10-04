////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6)
#define HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>

namespace hpx { namespace agas 
{

struct primary_namespace :
    components::client_base<primary_namespace, stubs::primary_namespace>
{
    // {{{ nested types 
    typedef components::client_base<primary_namespace, stubs::primary_namespace>
        base_type; 

    typedef server::primary_namespace server_type;

    typedef server_type::endpoint_type endpoint_type;
    typedef server_type::gva_type gva_type;
    typedef server_type::count_type count_type;
    typedef server_type::offset_type offset_type;
    typedef server_type::prefix_type prefix_type;
    // }}}

    primary_namespace()
      : base_type(bootstrap_primary_namespace_id())
    {}

    explicit primary_namespace(naming::id_type const& id)
      : base_type(id)
    {}

    ///////////////////////////////////////////////////////////////////////////
    // bind_locality and bind_gid interface 
    lcos::promise<response> bind_async(
        endpoint_type const& ep
      , count_type count = 0
        )
    {
        return this->base_type::bind_locality_async(this->gid_, ep, count);
    }

    response bind(
        endpoint_type const& ep
      , count_type count = 0
      , error_code& ec = throws
        )
    {
        return this->base_type::bind_locality(this->gid_, ep, count, ec);
    }

    response bind(
        endpoint_type const& ep
      , error_code& ec = throws
        )
    {
        return this->base_type::bind_locality(this->gid_, ep, 0, ec);
    }
    
    lcos::promise<response> bind_async(
        naming::gid_type const& gid
      , gva_type const& gva
        )
    {
        return this->base_type::bind_gid_async(this->gid_, gid, gva);
    }

    response bind(
        naming::gid_type const& gid
      , gva_type const& gva
      , error_code& ec = throws
        )
    {
        return this->base_type::bind_gid(this->gid_, gid, gva);
    }

    ///////////////////////////////////////////////////////////////////////////
    // page_fault interface 
    lcos::promise<response> page_fault_async(
        naming::gid_type const& gid
        )
    {
        return this->base_type::page_fault_async(this->gid_, gid);
    }
    
    response page_fault(
        naming::gid_type const& gid, error_code& ec = throws
        )
    {
        return this->base_type::page_fault(this->gid_, gid, ec);
    }
 
    ///////////////////////////////////////////////////////////////////////////
    // unbind_locality and unbind_gid interface 
    lcos::promise<response> unbind_async(
        endpoint_type const& ep
        )
    {
        return this->base_type::unbind_locality_async(this->gid_, ep);
    }
    
    response unbind(
        endpoint_type const& ep
      , error_code& ec = throws
        )
    {
        return this->base_type::unbind_locality(this->gid_, ep, ec);
    }

    lcos::promise<response> unbind_async(
        naming::gid_type const& gid
      , count_type count
        )
    {
        return this->base_type::unbind_gid_async(this->gid_, gid, count);
    }
    
    response unbind(
        naming::gid_type const& gid
      , count_type count
      , error_code& ec = throws
        )
    {
        return this->base_type::unbind_gid(this->gid_, gid, count, ec);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // increment interface 
    lcos::promise<response> increment_async(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return this->base_type::increment_async(this->gid_, gid, credits);
    }
    
    response increment(
        naming::gid_type const& gid
      , count_type credits
      , error_code& ec = throws
        )
    {
        return this->base_type::increment(this->gid_, gid, credits, ec);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // decrement interface 
    lcos::promise<response> decrement_async(
        naming::gid_type const& gid
      , count_type credits
        )
    {
        return this->base_type::decrement_async(this->gid_, gid, credits);
    }
    
    response decrement(
        naming::gid_type const& gid
      , count_type credits
      , error_code& ec = throws
        )
    {
        return this->base_type::decrement(this->gid_, gid, credits, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // localities interface 
    lcos::promise<response> localities_async()
    {
        return this->base_type::localities_async(this->gid_);
    }
    
    response localities(
        error_code& ec = throws
        )
    {
        return this->base_type::localities(this->gid_, ec);
    }
}; 

}}

#endif // HPX_389E034F_3BC6_4E6D_928B_B6E3088A54C6

