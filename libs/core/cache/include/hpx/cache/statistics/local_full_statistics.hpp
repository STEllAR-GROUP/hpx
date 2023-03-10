//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/cache/statistics/local_statistics.hpp>

#if defined(__bgq__)
#include <hwi/include/bqc/A2_inlines.h>
#endif

#include <chrono>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::util::cache::statistics {

    ///////////////////////////////////////////////////////////////////////////
    class local_full_statistics : public local_statistics
    {
    private:
        [[nodiscard]] static std::int64_t get_and_reset_value(
            std::int64_t& value, bool reset) noexcept
        {
            std::int64_t const result = value;
            if (reset)
                value = 0;
            return result;
        }

        struct api_counter_data
        {
            api_counter_data() = default;

            std::int64_t count_ = 0;
            std::int64_t time_ = 0;
        };

    public:
        /// Helper class to update timings and counts on function exit
        struct update_on_exit
        {
        private:
            [[nodiscard]] static constexpr api_counter_data&
            get_api_counter_data(local_full_statistics& stat, method m) noexcept
            {
                switch (m)
                {
                case method::get_entry:
                    [[fallthrough]];
                default:
                    break;

                case method::insert_entry:
                    return stat.insert_entry_;

                case method::update_entry:
                    return stat.update_entry_;

                case method::erase_entry:
                    return stat.erase_entry_;
                }

                return stat.get_entry_;
            }

            [[nodiscard]] static std::int64_t now() noexcept
            {
#if defined(__bgq__)
                return std::uint64_t(GetTimeBase());
#else
                std::chrono::nanoseconds const ns =
                    std::chrono::steady_clock::now().time_since_epoch();
                return static_cast<std::int64_t>(ns.count());
#endif
            }

        public:
            update_on_exit(local_full_statistics& stat, method m) noexcept
              : started_at_(now())
              , data_(get_api_counter_data(stat, m))
            {
            }

            ~update_on_exit()
            {
                data_.time_ += (now() - started_at_);
                ++data_.count_;
            }

            std::int64_t started_at_;
            api_counter_data& data_;
        };

        /// The function \a get_get_entry_count returns the number of
        /// invocations of the get_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_get_entry_count(bool reset) noexcept
        {
            return get_and_reset_value(get_entry_.count_, reset);
        }

        /// The function \a get_insert_entry_count returns the number of
        /// invocations of the insert_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_insert_entry_count(bool reset) noexcept
        {
            return get_and_reset_value(insert_entry_.count_, reset);
        }

        /// The function \a get_update_entry_count returns the number of
        /// invocations of the update_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_update_entry_count(bool reset) noexcept
        {
            return get_and_reset_value(update_entry_.count_, reset);
        }

        /// The function \a get_erase_entry_count returns the number of
        /// invocations of the erase() API function of the cache.
        [[nodiscard]] std::int64_t get_erase_entry_count(bool reset) noexcept
        {
            return get_and_reset_value(erase_entry_.count_, reset);
        }

        /// The function \a get_get_entry_time returns the overall time spent
        /// executing of the get_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_get_entry_time(bool reset) noexcept
        {
            return get_and_reset_value(get_entry_.time_, reset);
        }

        /// The function \a get_insert_entry_time returns the overall time
        /// spent executing of the insert_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_insert_entry_time(bool reset) noexcept
        {
            return get_and_reset_value(insert_entry_.time_, reset);
        }

        /// The function \a get_update_entry_time returns the overall time
        /// spent executing of the update_entry() API function of the cache.
        [[nodiscard]] std::int64_t get_update_entry_time(bool reset) noexcept
        {
            return get_and_reset_value(update_entry_.time_, reset);
        }

        /// The function \a get_erase_entry_time returns the overall time spent
        /// executing of the erase() API function of the cache.
        [[nodiscard]] std::int64_t get_erase_entry_time(bool reset) noexcept
        {
            return get_and_reset_value(erase_entry_.time_, reset);
        }

    private:
        friend struct update_on_exit;

        api_counter_data get_entry_;
        api_counter_data insert_entry_;
        api_counter_data update_entry_;
        api_counter_data erase_entry_;
    };
}    // namespace hpx::util::cache::statistics
