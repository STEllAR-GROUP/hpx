//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/debugging/print.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <iosfwd>

// ------------------------------------------------------------
/// \cond NODETAIL
namespace hpx::debug {

    // safely dump thread pointer/description
    template <typename T>
    struct threadinfo;

    // safely dump thread pointer/description
    template <>
    struct threadinfo<threads::thread_data*>
    {
        explicit constexpr threadinfo(threads::thread_data const* v) noexcept
          : data(v)
        {
        }

        threads::thread_data const* data;

        HPX_CORE_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<threads::thread_id_type*>
    {
        explicit constexpr threadinfo(threads::thread_id_type const* v) noexcept
          : data(v)
        {
        }

        threads::thread_id_type const* data;

        HPX_CORE_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<threads::thread_id_ref_type*>
    {
        explicit constexpr threadinfo(
            threads::thread_id_ref_type const* v) noexcept
          : data(v)
        {
        }

        threads::thread_id_ref_type const* data;

        HPX_CORE_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };

    template <>
    struct threadinfo<hpx::threads::thread_init_data>
    {
        explicit constexpr threadinfo(
            hpx::threads::thread_init_data const& v) noexcept
          : data(v)
        {
        }

        hpx::threads::thread_init_data const& data;

        HPX_CORE_EXPORT friend std::ostream& operator<<(
            std::ostream& os, threadinfo const& d);
    };
}    // namespace hpx::debug
/// \endcond
