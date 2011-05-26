////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_0D2D4035_D3B8_4E46_9E91_D0F0406D92D4)
#define HPX_0D2D4035_D3B8_4E46_9E91_D0F0406D92D4

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/component_base.hpp>
#include <hpx/runtime/agas/namespace/server/user_component.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database> 
struct user_component_namespace : component_namespace_base<
    components::stubs::stub_base<server::user_component_namespace<Database> >,
    server::user_component_namespace<Database>
> { };

}}}

#endif // HPX_0D2D4035_D3B8_4E46_9E91_D0F0406D92D4

