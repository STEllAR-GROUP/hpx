//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_FWD_HPP
#define HPX_RUNTIME_ACTIONS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>

#include <cstdint>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    struct base_action;

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    template <typename Action>
    struct transfer_action;

    template <typename Action>
    struct transfer_continuation_action;
#endif

    template <typename Component, typename Signature, typename Derived>
    struct basic_action;

    namespace detail
    {
        HPX_API_EXPORT std::uint32_t get_action_id_from_name(
            char const* action_name);
    }

    /// The type of an action defines whether this action will be executed
    /// directly or by an HPX-threads
    enum class action_flavor
    {
        plain_action = 0, ///< The action will be executed by a newly created thread
        direct_action = 1 ///< The action needs to be executed directly
    };

    /// \endcond
}}

#endif

