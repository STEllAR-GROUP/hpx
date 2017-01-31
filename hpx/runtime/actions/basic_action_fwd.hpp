//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP
#define HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP

#include <hpx/config.hpp>
#if defined(HPX_HAVE_ITTNOTIFY) && !defined(HPX_HAVE_APEX)
#include <hpx/util/itt_notify.hpp>
#endif

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Signature         return type and arguments
    /// \tparam Derived           derived action class
    template <typename Component, typename Signature, typename Derived>
    struct basic_action;

    //////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Action>
        char const* get_action_name();

#if defined(HPX_HAVE_ITTNOTIFY) && !defined(HPX_HAVE_APEX)
        template <typename Action>
        util::itt::string_handle const& get_action_name_itt();
#endif
    }
}}

#endif /*HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP*/
