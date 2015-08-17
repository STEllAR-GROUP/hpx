//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP)
#define HPX_RUNTIME_ACTIONS_BASIC_ACTION_FWD_HPP

#include <hpx/config.hpp>

namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    /// \tparam Component         component type
    /// \tparam Signature         return type and arguments
    /// \tparam Derived           derived action class
    template <typename Component, typename Signature, typename Derived>
    struct basic_action;
}}

#endif
