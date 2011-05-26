////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_C0775B40_D93B_4164_9F39_54A03939E3F4)
#define HPX_C0775B40_D93B_4164_9F39_54A03939E3F4

#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/symbol_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT bootstrap_symbol_namespace
  : symbol_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
            bootstrap_symbol_namespace<Database> >,
        Database>
{
    typedef symbol_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_SYMBOL_NS_MSB, HPX_AGAS_SYMBOL_NS_LSB, // constant GID
            bootstrap_symbol_namespace<Database> >,
        Database
    > base_type;

    bootstrap_symbol_namespace():
      base_type("bootstrap_symbol_namespace") {} 
};

}}}

#endif // HPX_C0775B40_D93B_4164_9F39_54A03939E3F4

