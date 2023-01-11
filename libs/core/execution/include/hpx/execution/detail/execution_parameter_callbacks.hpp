//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/topology/topology.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::execution::detail {

    /// \cond NOINTERNAL
    using get_os_thread_count_type = hpx::function<std::size_t()>;
    HPX_CORE_EXPORT void set_get_os_thread_count(get_os_thread_count_type f);
    HPX_CORE_EXPORT std::size_t get_os_thread_count();

    using get_pu_mask_type =
        hpx::function<threads::mask_type(threads::topology&, std::size_t)>;
    HPX_CORE_EXPORT void set_get_pu_mask(get_pu_mask_type f);
    HPX_CORE_EXPORT threads::mask_type get_pu_mask(
        threads::topology&, std::size_t);
    /// \endcond
}    // namespace hpx::parallel::execution::detail
