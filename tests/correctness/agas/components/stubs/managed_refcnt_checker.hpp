////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE)
#define HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <tests/correctness/agas/components/server/managed_refcnt_checker.hpp>

namespace hpx { namespace test { namespace stubs
{

struct managed_refcnt_checker
  : components::stub_base<server::managed_refcnt_checker>
{

};            

}}}

#endif // HPX_FCB1AFA8_8399_40D9_95DE_A68F861C0CFE

