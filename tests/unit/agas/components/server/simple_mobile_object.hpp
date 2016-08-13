////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_DB9FD2D1_DFD8_491C_AB4B_1CDBCC56A7D0)
#define HPX_DB9FD2D1_DFD8_491C_AB4B_1CDBCC56A7D0

#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <cstdint>

namespace hpx { namespace test { namespace server
{

struct HPX_COMPONENT_EXPORT simple_mobile_object
  : components::simple_component_base<simple_mobile_object>
{
  public:
    simple_mobile_object() {}

    std::uint64_t get_lva()
    {
        return reinterpret_cast<std::uint64_t>(this);
    }

    HPX_DEFINE_COMPONENT_ACTION(simple_mobile_object
                              , get_lva
                              , get_lva_action);
};

}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::test::server::simple_mobile_object::get_lva_action,
    simple_mobile_object_get_lva_action);

#endif // HPX_DB9FD2D1_DFD8_491C_AB4B_1CDBCC56A7D0

