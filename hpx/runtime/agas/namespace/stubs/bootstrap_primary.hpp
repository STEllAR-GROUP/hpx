////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_72235680_3DD1_4AFB_A34D_ACB968370B45)
#define HPX_72235680_3DD1_4AFB_A34D_ACB968370B45

#include <hpx/runtime/agas/namespace/stubs/primary_base.hpp>
#include <hpx/runtime/agas/namespace/server/bootstrap_primary.hpp>

namespace hpx { namespace agas { namespace stubs
{

template <typename Database, typename Protocol> 
struct bootstrap_primary_namespace : primary_namespace_base<
    empty, server::bootstrap_primary_namespace<Database, Protocol>
> { };

}}}

#endif // HPX_72235680_3DD1_4AFB_A34D_ACB968370B45

