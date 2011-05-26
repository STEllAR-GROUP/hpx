////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_EFAA8AC3_58E6_4B5F_95B7_8AD6DC836357)
#define HPX_EFAA8AC3_58E6_4B5F_95B7_8AD6DC836357

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/primary_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database, typename Protocol>
struct HPX_COMPONENT_EXPORT user_primary_namespace
  : primary_namespace_base<
        components::simple_component_base<
            user_primary_namespace<Database, Protocol> >,
        Database, Protocol>
{
    typedef primary_namespace_base<
        components::simple_component_base<
            user_primary_namespace<Database, Protocol> >,
        Database, Protocol
    > base_type;

    user_primary_namespace(): base_type("user_primary_namespace") {} 
};

}}}

#endif // HPX_EFAA8AC3_58E6_4B5F_95B7_8AD6DC836357

