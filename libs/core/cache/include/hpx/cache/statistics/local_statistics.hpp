//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/cache/statistics/no_statistics.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::cache::statistics {

    ///////////////////////////////////////////////////////////////////////////
    class local_statistics : public no_statistics
    {
    public:
        local_statistics() = default;

        std::size_t get_and_reset(std::size_t& value, bool reset) noexcept
        {
            std::size_t result = value;
            if (reset)
                value = 0;
            return result;
        }

        constexpr std::size_t hits() const noexcept
        {
            return hits_;
        }
        constexpr std::size_t misses() const noexcept
        {
            return misses_;
        }
        constexpr std::size_t insertions() const noexcept
        {
            return insertions_;
        }
        constexpr std::size_t evictions() const noexcept
        {
            return evictions_;
        }

        std::size_t hits(bool reset) noexcept
        {
            return get_and_reset(hits_, reset);
        }
        std::size_t misses(bool reset) noexcept
        {
            return get_and_reset(misses_, reset);
        }
        std::size_t insertions(bool reset) noexcept
        {
            return get_and_reset(insertions_, reset);
        }
        std::size_t evictions(bool reset) noexcept
        {
            return get_and_reset(evictions_, reset);
        }

        /// \brief  The function \a got_hit will be called by a cache instance
        ///         whenever a entry got touched.
        void got_hit() noexcept
        {
            ++hits_;
        }

        /// \brief  The function \a got_miss will be called by a cache instance
        ///         whenever a requested entry has not been found in the cache.
        void got_miss() noexcept
        {
            ++misses_;
        }

        /// \brief  The function \a got_insertion will be called by a cache
        ///         instance whenever a new entry has been inserted.
        void got_insertion() noexcept
        {
            ++insertions_;
        }

        /// \brief  The function \a got_eviction will be called by a cache
        ///         instance whenever an entry has been removed from the cache
        ///         because a new inserted entry let the cache grow beyond its
        ///         capacity.
        void got_eviction() noexcept
        {
            ++evictions_;
        }

        /// \brief Reset all statistics
        void clear() noexcept
        {
            hits_ = 0;
            misses_ = 0;
            evictions_ = 0;
            insertions_ = 0;
        }

    private:
        std::size_t hits_ = 0;
        std::size_t misses_ = 0;
        std::size_t insertions_ = 0;
        std::size_t evictions_ = 0;
    };
}    // namespace hpx::util::cache::statistics
