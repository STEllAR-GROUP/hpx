////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_9F654816_5D8C_4A0E_B17B_DF5D0685FE50)
#define HPX_9F654816_5D8C_4A0E_B17B_DF5D0685FE50

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/agas/namespace/server/symbol_base.hpp>

namespace hpx { namespace agas { namespace server
{

template <typename Database>
struct HPX_COMPONENT_EXPORT user_symbol_namespace
  : symbol_namespace_base<
        components::simple_component_base<user_symbol_namespace<Database> >,
        Database>
{
    typedef symbol_namespace_base<
        components::simple_component_base<user_symbol_namespace<Database> >,
        Database
    > base_type;

    user_symbol_namespace(): base_type("user_symbol_namespace") {} 
};

}}}

#endif // HPX_9F654816_5D8C_4A0E_B17B_DF5D0685FE50

