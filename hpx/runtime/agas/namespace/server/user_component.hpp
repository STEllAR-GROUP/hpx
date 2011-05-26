////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B4858C4F_4AD9_444B_A6E0_87CCFE77A9B7)
#define HPX_B4858C4F_4AD9_444B_A6E0_87CCFE77A9B7

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/component_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT user_component_namespace
  : component_namespace_base<
        components::simple_component_base<user_component_namespace<Database> >,
        Database>
{
    typedef component_namespace_base<
        components::simple_component_base<user_component_namespace<Database> >,
        Database
    > base_type;

    user_component_namespace(): base_type("user_component_namespace") {} 
};

}}}

#endif // HPX_B4858C4F_4AD9_444B_A6E0_87CCFE77A9B7

