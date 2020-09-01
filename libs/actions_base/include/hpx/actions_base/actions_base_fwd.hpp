//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

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

    /// The \a base_action class is an abstract class used as the base class
    /// for all action types. It's main purpose is to allow polymorphic
    /// serialization of action instances through a unique_ptr.
    struct HPX_EXPORT base_action;

    namespace detail {

        HPX_EXPORT std::uint32_t get_action_id_from_name(
            char const* action_name);
    }    // namespace detail

    /// \endcond
}}    // namespace hpx::actions
