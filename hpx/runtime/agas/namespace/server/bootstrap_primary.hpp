////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_3C8E2C8F_C61E_48C3_A782_E45570A6A287)
#define HPX_3C8E2C8F_C61E_48C3_A782_E45570A6A287

#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/primary_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct HPX_COMPONENT_EXPORT bootstrap_primary_namespace
  : primary_namespace_base<
       components::fixed_component_base<
            HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
            bootstrap_primary_namespace<Database, Protocol> >,
        Database, Protocol>
{
    typedef primary_namespace_base<
       components::fixed_component_base<
            HPX_AGAS_PRIMARY_NS_MSB, HPX_AGAS_PRIMARY_NS_LSB, // constant GID
            bootstrap_primary_namespace<Database, Protocol> >,
        Database, Protocol
    > base_type;

    bootstrap_primary_namespace():
      base_type("bootstrap_primary_namespace") {} 
};

}}}

#endif // HPX_3C8E2C8F_C61E_48C3_A782_E45570A6A287

