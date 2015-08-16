//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLY_IMPLEMENTATIONS_FWD_APR_13_2015_0945AM)
#define HPX_APPLY_IMPLEMENTATIONS_FWD_APR_13_2015_0945AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>

namespace hpx { namespace applier { namespace detail
{
    // forward declaration only
    template <typename Action, typename ...Ts>
    bool apply_l_p(actions::continuation_type const& c,
        naming::id_type const& target, naming::address&& addr,
        threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename ...Ts>
    bool apply_r_p(naming::address&& addr, actions::continuation_type const& c,
        naming::id_type const& id, threads::thread_priority priority,
        Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    bool apply_r_p_cb(naming::address&& addr,
        actions::continuation_type const& c, naming::id_type const& id,
        threads::thread_priority priority, Callback && cb, Ts&&... vs);
}}}

namespace hpx { namespace detail
{
    template <typename Action, typename ...Ts>
    bool apply_impl(actions::continuation_type const& c,
        hpx::id_type const& id, threads::thread_priority priority, Ts&&... vs);

    template <typename Action, typename Callback, typename ...Ts>
    bool apply_cb_impl(actions::continuation_type const& c,
        hpx::id_type const& id, threads::thread_priority priority,
        Callback&& cb, Ts&&... vs);
}}

#endif
