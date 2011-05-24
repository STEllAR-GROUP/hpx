////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E5F37926_4705_48AE_AE28_844E5ED05B9F)
#define HPX_E5F37926_4705_48AE_AE28_844E5ED05B9F

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/util/runtime_configuration.hpp>

namespace hpx { namespace agas { namespace legacy
{

// TODO: pass error codes once they're implemented in AGAS.
template <typename Database>
struct bootstrap_agent
{
    // {{{ types
    typedef bootstrap_primary_namespace<Database, tag::network::tcpip>
        primary_namespace_type;

    typedef bootstrap_component_namespace<Database> component_namespace_type;
    typedef bootstrap_symbol_namespace<Database> symbol_namespace_type;

    typedef typename component_namespace_type::component_id_type
        component_id_type;
    
    typedef typename primary_namespace_type::gva_type gva_type;
    typedef typename primary_namespace_type::count_type count_type;
    typedef typename primary_namespace_type::offset_type offset_type;
    typedef typename primary_namespace_type::endpoint_type endpoint_type;
    typedef typename primary_namespace_type::unbinding_type unbinding_type;
    typedef typename primary_namespace_type::binding_type binding_type;
    typedef typename component_namespace_type::prefixes_type prefixes_type;
    typedef typename component_namespace_type::prefix_type prefix_type;
    typedef typename primary_namespace_type::decrement_type decrement_type;
    // }}}

  protected:
    primary_namespace_type primary_ns_;
    component_namespace_type component_ns_;
    symbol_namespace_type symbol_ns_;

    bootstrap_agent_base(util::runtime_configuration const& ini_,
                         runtime_mode mode)
    {
        // IMPLEMENT     
    }
};

}}}

#endif // HPX_E5F37926_4705_48AE_AE28_844E5ED05B9F

