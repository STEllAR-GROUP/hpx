//  Copyright (c)      2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_ACTIONS_FWD_HPP
#define HPX_RUNTIME_ACTIONS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/actions/continuation_fwd.hpp>

namespace hpx { namespace actions
{
    /// \cond NOINTERNAL

    struct base_action;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct transfer_action;

    template <typename Action>
    struct transfer_continuation_action;

    template <typename Component, typename Signature, typename Derived>
    struct basic_action;
}}

#endif

