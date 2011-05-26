////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_302B7AB3_6CF8_4199_83DC_AB78C2BFA359)
#define HPX_302B7AB3_6CF8_4199_83DC_AB78C2BFA359

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/agas/namespace/stubs/primary_base.hpp>
#include <hpx/runtime/agas/namespace/server/user_primary.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol> 
struct user_primary_namespace : primary_namespace_base<
    components::stubs::stub_base<
        server::user_primary_namespace<Database, Protocol>
    >,
    server::user_primary_namespace<Database, Protocol>
> { };

}}}

#endif // HPX_302B7AB3_6CF8_4199_83DC_AB78C2BFA359

