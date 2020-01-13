//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_ASYNC_UNWRAP_IMPLEMENTATIONS_FWD_JUL_22_2018_0145PM)
#define HPX_LCOS_ASYNC_UNWRAP_IMPLEMENTATIONS_FWD_JUL_22_2018_0145PM

#include <hpx/config.hpp>
#include <hpx/lcos/async_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/extract_action.hpp>

namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Launch, typename ...Ts>
    typename hpx::traits::extract_action<Action>::type::local_result_type
    async_unwrap_result_impl(Launch && policy, hpx::id_type const& id, Ts&&... vs);
}}

#endif
