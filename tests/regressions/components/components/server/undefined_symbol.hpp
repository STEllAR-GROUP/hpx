////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_BDD6649D_AA26_435E_96D8_07C9B733E272)
#define HPX_BDD6649D_AA26_435E_96D8_07C9B733E272

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace test { namespace server
{

struct undefined_symbol : components::managed_component_base<undefined_symbol>
{
    void break_();

    enum actions
    {
        action_break
    };

    typedef hpx::actions::action0<
        // component
        undefined_symbol
        // action code
      , action_break
        // method
      , &undefined_symbol::break_
    > break_action;
};

}}}

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::test::server::undefined_symbol::break_action
  , test_undefined_symbol_break_action);

#endif // HPX_BDD6649D_AA26_435E_96D8_07C9B733E272

