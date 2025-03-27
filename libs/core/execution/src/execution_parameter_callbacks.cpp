//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2017-2-24 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/execution/detail/execution_parameter_callbacks.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::execution::detail {

    get_os_thread_count_type& get_get_os_thread_count()
    {
        static get_os_thread_count_type f;
        return f;
    }

    void set_get_os_thread_count(get_os_thread_count_type f)
    {
        get_get_os_thread_count() = HPX_MOVE(f);
    }

    std::size_t get_os_thread_count()
    {
        if (get_get_os_thread_count())
        {
            return get_get_os_thread_count()();
        }

        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
            "hpx::parallel::execution::detail::get_os_thread_count",
            "No fallback handler for get_os_thread_count is installed. "
            "Please start the runtime if you haven't done so. If you "
            "intended to not use the runtime make sure you have "
            "implemented get_os_thread_count for your executor or install "
            "a fallback handler with "
            "hpx::parallel::execution::detail::set_get_os_thread_count.");
    }

    get_pu_mask_type& get_get_pu_mask()
    {
        static get_pu_mask_type f;
        return f;
    }

    void set_get_pu_mask(get_pu_mask_type f)
    {
        get_get_pu_mask() = HPX_MOVE(f);
    }

    threads::mask_type get_pu_mask(
        threads::topology& topo, std::size_t thread_num)
    {
        if (get_get_pu_mask())
        {
            return get_get_pu_mask()(topo, thread_num);
        }

        HPX_THROW_EXCEPTION(hpx::error::invalid_status,
            "hpx::parallel::execution::detail::get_pu_mask",
            "No fallback handler for get_pu_mask is installed. Please "
            "start the runtime if you haven't done so. If you intended "
            "to not use the runtime make sure you have implemented "
            "get_pu_mask for your executor or install a fallback "
            "handler with "
            "hpx::parallel::execution::detail::set_get_pu_mask.");
    }
}    // namespace hpx::parallel::execution::detail
