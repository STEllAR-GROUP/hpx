//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SYNC_IMPLEMENTATIONS_FWD_JUL_21_2018_0917PM)
#define HPX_LCOS_SYNC_IMPLEMENTATIONS_FWD_JUL_21_2018_0917PM

#include <hpx/config.hpp>
#include <hpx/lcos/sync_fwd.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/traits/extract_action.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    sync_impl(launch policy, hpx::id_type const& id, Ts&&... vs);

    template <typename Action, typename... Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    sync_impl(hpx::detail::sync_policy, hpx::id_type const& id, Ts&&... vs);
}}

#endif
