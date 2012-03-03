////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_237CB566_DEBF_4762_9776_24B2B7ECE61D)
#define HPX_237CB566_DEBF_4762_9776_24B2B7ECE61D

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <tests/regressions/components/components/server/undefined_symbol.hpp>

namespace hpx { namespace test { namespace stubs
{

struct undefined_symbol : components::stub_base<server::undefined_symbol> { };

}}}

#endif // HPX_237CB566_DEBF_4762_9776_24B2B7ECE61D

