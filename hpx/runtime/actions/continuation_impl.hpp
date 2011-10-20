//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_CONTINUATION_IMPL_JUN_20_2008_0851PM)
#define HPX_RUNTIME_ACTIONS_CONTINUATION_IMPL_JUN_20_2008_0851PM

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/lcos/base_lco.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Arg0>
    inline Arg0 const&
    continuation::trigger(Arg0 const& arg0)
    {
        typedef typename
            lcos::template base_lco_with_value<Arg0>::set_result_action
        action_type;

        LLCO_(info) << "promise::set(" << gid_ << ")";

        applier::apply<action_type>(gid_, arg0);
        return arg0;
    }

}}

#endif
