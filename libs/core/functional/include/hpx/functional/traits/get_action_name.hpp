//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c)      2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if (HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX))
#include <hpx/modules/itt_notify.hpp>
#endif

namespace hpx::actions::detail {

    template <typename Action>
    [[nodiscard]] char const* get_action_name() noexcept;

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Action>
    [[nodiscard]] util::itt::string_handle const&
    get_action_name_itt() noexcept;
#endif
}    // namespace hpx::actions::detail
