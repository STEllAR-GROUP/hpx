//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/util.hpp>

#include <hpx/parcelset_base/detail/data_point.hpp>

#include <cstdint>
#include <mutex>

namespace hpx::parcelset {

    namespace detail {

        /// Collect statistics information about parcels sent and received.
        template <typename Mutex>
        class gatherer
        {
        public:
            gatherer() = default;

            void add_data(data_point const& x);

            inline std::int64_t num_parcels(bool reset);
            inline std::int64_t num_messages(bool reset);
            inline std::int64_t total_bytes(bool reset);
            inline std::int64_t total_raw_bytes(bool reset);
            inline std::int64_t total_time(bool reset);
            inline std::int64_t total_serialization_time(bool reset);
            inline std::int64_t total_buffer_allocate_time(bool reset);
            inline std::int64_t num_zchunks(bool reset);
            inline std::int64_t num_zchunks_per_msg_max(bool reset);
            inline std::int64_t size_zchunks_total(bool reset);
            inline std::int64_t size_zchunks_max(bool reset);

        private:
            std::int64_t overall_bytes_ = 0;
            std::int64_t overall_time_ = 0;
            std::int64_t serialization_time_ = 0;
            std::int64_t num_parcels_ = 0;
            std::int64_t num_messages_ = 0;
            std::int64_t overall_raw_bytes_ = 0;
            std::int64_t buffer_allocate_time_ = 0;
            std::int64_t num_zchunks_ = 0;
            std::int64_t num_zchunks_per_msg_max_ = 0;
            std::int64_t size_zchunks_total_ = 0;
            std::int64_t size_zchunks_max_ = 0;

            // Create mutex for accumulator functions.
            Mutex acc_mtx;
        };

        template <typename Mutex>
        void gatherer<Mutex>::add_data(data_point const& x)
        {
            std::lock_guard l(acc_mtx);

            overall_bytes_ += x.bytes_;
            overall_time_ += x.time_;
            serialization_time_ += x.serialization_time_;
            num_parcels_ += x.num_parcels_;
            overall_raw_bytes_ += x.raw_bytes_;
            ++num_messages_;
            buffer_allocate_time_ += x.buffer_allocate_time_;
            num_zchunks_ += x.num_zchunks_;
            num_zchunks_per_msg_max_ = (std::max)(
                num_zchunks_per_msg_max_, x.num_zchunks_per_msg_max_);
            size_zchunks_total_ += x.size_zchunks_total_;
            size_zchunks_max_ =
                (std::max)(size_zchunks_max_, x.size_zchunks_max_);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::num_parcels(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(num_parcels_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::num_messages(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(num_messages_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::total_time(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(overall_time_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::total_serialization_time(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(serialization_time_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::total_bytes(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(overall_bytes_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::total_raw_bytes(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(overall_raw_bytes_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::total_buffer_allocate_time(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(buffer_allocate_time_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::num_zchunks(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(num_zchunks_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::num_zchunks_per_msg_max(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(num_zchunks_per_msg_max_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::size_zchunks_total(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(size_zchunks_total_, reset);
        }

        template <typename Mutex>
        std::int64_t gatherer<Mutex>::size_zchunks_max(bool reset)
        {
            std::lock_guard l(acc_mtx);
            return util::get_and_reset_value(size_zchunks_max_, reset);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    using gatherer = detail::gatherer<hpx::spinlock>;
    using gatherer_nolock = detail::gatherer<hpx::no_mutex>;
}    // namespace hpx::parcelset
