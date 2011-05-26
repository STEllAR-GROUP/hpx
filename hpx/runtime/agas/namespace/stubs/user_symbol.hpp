////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_09C8865C_0908_48B1_8833_700E98A6E326)
#define HPX_09C8865C_0908_48B1_8833_700E98A6E326

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/symbol_base.hpp>
#include <hpx/runtime/agas/namespace/server/user_symbol.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database> 
struct user_symbol_namespace : symbol_namespace_base<
    components::stubs::stub_base<server::user_symbol_namespace<Database> >,
    server::user_symbol_namespace<Database>
> { };

}}}

#endif // HPX_09C8865C_0908_48B1_8833_700E98A6E326

