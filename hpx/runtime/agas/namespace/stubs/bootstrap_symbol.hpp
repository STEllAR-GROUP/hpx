////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_7DF6A23B_4D8D_4239_95A6_EBC6BA338680)
#define HPX_7DF6A23B_4D8D_4239_95A6_EBC6BA338680

#include <hpx/runtime/agas/namespace/stubs/symbol_base.hpp>
#include <hpx/runtime/agas/namespace/server/bootstrap_symbol.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database> 
struct bootstrap_symbol_namespace : symbol_namespace_base<
    empty, server::bootstrap_symbol_namespace<Database>
> { };

}}}

#endif // HPX_7DF6A23B_4D8D_4239_95A6_EBC6BA338680

