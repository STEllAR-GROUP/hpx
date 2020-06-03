//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_NETWORKING) &&                                            \
    (HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX))
#include <hpx/modules/itt_notify.hpp>
#endif

namespace hpx { namespace actions { namespace detail {
    template <typename Action>
    char const* get_action_name();

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Action>
    util::itt::string_handle const& get_action_name_itt();
#endif
}}}    // namespace hpx::actions::detail
