//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP
#define HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP

#include <hpx/config.hpp>
#if defined(HPX_HAVE_NETWORKING) &&                                            \
    (HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX))
#include <hpx/concurrency/itt_notify.hpp>
#endif
#include <hpx/functional/traits/get_action_name.hpp>
#include <hpx/runtime/actions/preassigned_action_id.hpp>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Signature         return type and arguments
    /// \tparam Derived           derived action class
    template <typename Component, typename Signature, typename Derived>
    struct basic_action;
}}

#endif /*HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP*/
