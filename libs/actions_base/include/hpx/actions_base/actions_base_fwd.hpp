//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace actions {

    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Signature, typename Derived>
    struct basic_action;

    /// The type of an action defines whether this action will be executed
    /// directly or by an HPX-threads
    enum class action_flavor
    {
        plain_action = 0,    ///< The action will be executed by a newly created
        ///< thread
        direct_action = 1    ///< The action needs to be executed directly
    };

    /// \endcond
}}    // namespace hpx::actions
