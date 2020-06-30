//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file action_support.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/actions_fwd.hpp>
#include <hpx/actions_base/action_base_support.hpp>
#include <hpx/actions_base/traits/action_remote_result.hpp>
#include <hpx/debugging/demangle_helper.hpp>
#include <hpx/serialization/base_object.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <hpx/modules/itt_notify.hpp>
#endif

#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
/// \namespace actions
namespace hpx { namespace actions { namespace detail {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_NETWORKING)
    template <typename Action>
    char const* get_action_name()
#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
        ;
#else
    {
        /// If you encounter this assert while compiling code, that means that
        /// you have a HPX_REGISTER_ACTION macro somewhere in a source file,
        /// but the header in which the action is defined misses a
        /// HPX_REGISTER_ACTION_DECLARATION
        static_assert(traits::needs_automatic_registration<Action>::value,
            "HPX_REGISTER_ACTION_DECLARATION missing");
        return util::debug::type_id<Action>::typeid_.type_id();
    }
#endif
#else    // HPX_HAVE_NETWORKING
    template <typename Action>
    char const* get_action_name()
    {
        return util::debug::type_id<Action>::typeid_.type_id();
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    std::uint32_t get_action_id()
    {
        static std::uint32_t id =
            get_action_id_from_name(get_action_name<Action>());
        return id;
    }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Action>
    util::itt::string_handle const& get_action_name_itt()
#if !defined(HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION)
        ;
#else
    {
        static util::itt::string_handle sh = get_action_name<Action>();
        return sh;
    }
#endif
#endif

    /// \endcond
}}}    // namespace hpx::actions::detail

#include <hpx/config/warnings_suffix.hpp>
