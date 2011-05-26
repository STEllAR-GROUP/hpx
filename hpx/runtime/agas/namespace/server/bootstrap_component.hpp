////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_D075777E_924E_4FE5_9CF2_CDEFFB538310)
#define HPX_D075777E_924E_4FE5_9CF2_CDEFFB538310

#include <hpx/runtime/components/server/fixed_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT bootstrap_component_namespace
  : component_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
            bootstrap_component_namespace<Database> >,
        Database>
{
    typedef component_namespace_base<
        components::fixed_component_base<
            HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB, // constant GID
            bootstrap_component_namespace<Database> >,
        Database
    > base_type;

    bootstrap_component_namespace():
      base_type("bootstrap_component_namespace") {} 
};

}}}

#endif // HPX_D075777E_924E_4FE5_9CF2_CDEFFB538310

