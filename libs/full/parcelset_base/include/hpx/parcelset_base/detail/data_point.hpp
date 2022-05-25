//  Copyright (c) 2011 Katelyn Kufahl
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstddef>
#include <cstdint>

namespace hpx::parcelset {

    /// A \a data_point collects all timing and statistical information
    /// for a single parcel (either sent or received).
    struct data_point
    {
        /// number of bytes on the wire for this parcel (possibly compressed)
        std::size_t bytes_ = 0;

        /// during processing holds start timestamp after processing holds
        /// elapsed time
        std::int64_t time_ = 0;

        /// during processing holds start serialization timestamp after
        /// processing holds elapsed serialization time
        std::int64_t serialization_time_ = 0;

        /// The number of parcels processed by this message
        std::size_t num_parcels_ = 0;

        /// number of bytes processed for the action in this parcel (uncompressed)
        std::size_t raw_bytes_ = 0;

        /// The time spent for allocating buffers
        std::int64_t buffer_allocate_time_ = 0;

        //// number of zero-copy chunks in total
        std::int64_t num_zchunks_ = 0;

        //// maximum number of zero-copy chunks per message
        std::int64_t num_zchunks_per_msg_max_ = 0;

        //// size of zero-copy chunks in total
        std::int64_t size_zchunks_total_ = 0;

        //// maximum size of zero-copy chunks
        std::int64_t size_zchunks_max_ = 0;
    };
}    // namespace hpx::parcelset
