//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::cache::statistics {

    ///////////////////////////////////////////////////////////////////////////
    enum class method
    {
        get_entry = 0,
        insert_entry = 1,
        update_entry = 2,
        erase_entry = 3
    };

#define HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG                         \
    "The unscoped scheduler_mode names are deprecated. Please use "            \
    "scheduler_mode::state instead."

    HPX_DEPRECATED_V(1, 8, HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr method method_get_entry = method::get_entry;
    HPX_DEPRECATED_V(1, 8, HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr method method_insert_entry = method::insert_entry;
    HPX_DEPRECATED_V(1, 8, HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr method method_update_entry = method::update_entry;
    HPX_DEPRECATED_V(1, 8, HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG)
    inline constexpr method method_erase_entry = method::erase_entry;

#undef HPX_CACHE_METHOD_UNSCOPED_ENUM_DEPRECATION_MSG

    ///////////////////////////////////////////////////////////////////////////
    class no_statistics
    {
    public:
        /// \brief  The function \a got_hit will be called by a cache instance
        ///         whenever a entry got touched.
        constexpr void got_hit() const noexcept {}

        /// \brief  The function \a got_miss will be called by a cache instance
        ///         whenever a requested entry has not been found in the cache.
        constexpr void got_miss() const noexcept {}

        /// \brief  The function \a got_insertion will be called by a cache
        ///         instance whenever a new entry has been inserted.
        constexpr void got_insertion() const noexcept {}

        /// \brief  The function \a got_eviction will be called by a cache
        ///         instance whenever an entry has been removed from the cache
        ///         because a new inserted entry let the cache grow beyond its
        ///         capacity.
        constexpr void got_eviction() const noexcept {}

        /// \brief Reset all statistics
        constexpr void clear() const noexcept {}

        /// Helper class to update timings and counts on function exit
        struct update_on_exit
        {
            constexpr update_on_exit(no_statistics const&, method) noexcept {}
        };

        /// The function \a get_get_entry_count returns the number of
        /// invocations of the get_entry() API function of the cache.
        constexpr std::int64_t get_get_entry_count(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_insert_entry_count returns the number of
        /// invocations of the insert_entry() API function of the cache.
        constexpr std::int64_t get_insert_entry_count(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_update_entry_count returns the number of
        /// invocations of the update_entry() API function of the cache.
        constexpr std::int64_t get_update_entry_count(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_erase_entry_count returns the number of
        /// invocations of the erase() API function of the cache.
        constexpr std::int64_t get_erase_entry_count(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_get_entry_time returns the overall time spent
        /// executing of the get_entry() API function of the cache.
        constexpr std::int64_t get_get_entry_time(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_insert_entry_time returns the overall time
        /// spent executing of the insert_entry() API function of the cache.
        constexpr std::int64_t get_insert_entry_time(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_update_entry_time returns the overall time
        /// spent executing of the update_entry() API function of the cache.
        constexpr std::int64_t get_update_entry_time(bool) const noexcept
        {
            return 0;
        }

        /// The function \a get_erase_entry_time returns the overall time spent
        /// executing of the erase() API function of the cache.
        constexpr std::int64_t get_erase_entry_time(bool) const noexcept
        {
            return 0;
        }
    };
}    // namespace hpx::util::cache::statistics
