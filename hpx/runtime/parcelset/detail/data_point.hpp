////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <cstdint>

namespace hpx { namespace performance_counters { namespace parcels {
    /// \brief A \a data_point collects all timing and statistical information
    ///        for a single parcel (either sent or received).
    struct data_point
    {
        data_point()
          : bytes_(0)
          , time_(0)
          , serialization_time_(0)
          , num_parcels_(0)
          , raw_bytes_(0)
          , buffer_allocate_time_(0)
        {
        }

        std::size_t bytes_;    ///< number of bytes on the wire for this parcel
                               ///< (possibly compressed)
        std::int64_t time_;    ///< during processing holds start timestamp
                               ///< after processing holds elapsed time
        std::int64_t serialization_time_;    ///< during processing holds
            ///< start serialization timestamp after
            ///< processing holds elapsed serialization time
        std::size_t
            num_parcels_;    ///< The number of parcels processed by this message
        std::size_t
            raw_bytes_;    ///< number of bytes processed for the action in
                           ///< this parcel (uncompressed)

        std::int64_t
            buffer_allocate_time_;    ///< The time spent for allocating buffers
    };
}}}    // namespace hpx::performance_counters::parcels
