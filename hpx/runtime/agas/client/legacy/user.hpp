////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_65C0CE6B_B9BD_4F34_B627_57A5FD333630)
#define HPX_65C0CE6B_B9BD_4F34_B627_57A5FD333630

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/runtime/agas/namespace/component.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/runtime/agas/client/legacy/resolver_cache.hpp>

namespace hpx { namespace agas { namespace legacy
{

// TODO: pass error codes once they're implemented in AGAS.
template <typename Database>
struct user : resolver_cache<tag::network::tcpip>
{
  protected:
    // {{{ types
    typedef resolver_cache<tag::network::tcpip> base_type;

    typedef primary_namespace<Database, tag::network::tcpip>
        primary_namespace_type;

    typedef component_namespace<Database> component_namespace_type;
    typedef symbol_namespace<Database> symbol_namespace_type;
    // }}}

    primary_namespace_type primary_ns_;
    component_namespace_type component_ns_;
    symbol_namespace_type symbol_ns_;

    user(util::runtime_configuration const& ini_, runtime_mode)
        : base_type(ini_)
    {
        naming::id_type prefix
            = applier::get_applier().get_runtime_support_gid();
        primary_ns_.create(prefix);
        component_ns_.create(prefix);
        symbol_ns_.create(prefix);
    }
};

}}}

#endif // HPX_65C0CE6B_B9BD_4F34_B627_57A5FD333630

