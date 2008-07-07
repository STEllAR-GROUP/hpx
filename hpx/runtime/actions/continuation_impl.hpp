//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_IMPL_JUN_20_2008_0851PM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_IMPL_JUN_20_2008_0851PM

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/action.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/base_lco.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    inline 
    void continuation::trigger_all(applier::applier& app)
    {
        std::vector<naming::id_type>::iterator end = gids_.end();
        for (std::vector<naming::id_type>::iterator it = gids_.begin();
             it != end; ++it)
        {
            if (!app.apply<lcos::base_lco::set_event_action>(*it))
                break;
        }
    }

    template <typename Arg0>
    inline void 
    continuation::trigger_all(applier::applier& app, Arg0 const& arg0)
    {
        typedef typename 
            lcos::template base_lco_with_value<Arg0>::set_result_action 
        action_type;

        std::vector<naming::id_type>::iterator end = gids_.end();
        for (std::vector<naming::id_type>::iterator it = gids_.begin();
             it != end; ++it)
        {
            if (!app.apply<action_type>(*it, arg0))
                break;
        }
    }

}}

#endif
