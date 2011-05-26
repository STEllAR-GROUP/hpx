////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_EED83275_D3DB_4CFB_AC54_2DE4BE1FF4A5)
#define HPX_EED83275_D3DB_4CFB_AC54_2DE4BE1FF4A5

#include <hpx/runtime/agas/namespace/stubs/component_base.hpp>
#include <hpx/runtime/agas/namespace/server/bootstrap_component.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database> 
struct bootstrap_component_namespace : component_namespace_base<
    empty, server::bootstrap_component_namespace<Database>
> { };

}}}

#endif // HPX_EED83275_D3DB_4CFB_AC54_2DE4BE1FF4A5

