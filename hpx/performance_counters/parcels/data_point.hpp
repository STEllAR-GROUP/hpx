////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_76311D67_43DA_4B3A_8A2A_14B8A1A266D9)
#define HPX_76311D67_43DA_4B3A_8A2A_14B8A1A266D9

#include <hpx/runtime/naming/name.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace performance_counters { namespace parcels
{
    /// \brief A \a data_point collects all timing and statistical information
    ///        for a single parcel (either sent or received).
    struct data_point
    {
        data_point()
          : bytes_(0)
          , time_(0)
          , serialization_time_(0)
#if defined(HPX_HAVE_SECURITY)
          , security_time_(0)
#endif
          , num_parcels_(0)
          , raw_bytes_(0)
          , buffer_allocate_time_(0)
        {}

        std::size_t bytes_;        ///< number of bytes on the wire for this parcel
                                   ///< (possibly compressed)
        boost::int64_t time_;      ///< during processing holds start timestamp
                                   ///< after processing holds elapsed time
        boost::int64_t serialization_time_;    ///< during processing holds
                                   ///< start serialization timestamp after
                                   ///< processing holds elapsed serialization time
#if defined(HPX_HAVE_SECURITY)
        boost::int64_t security_time_;///< during processing this holds holds the start
                                   ///< security work timestamp after
                                   ///< processing holds elapsed security time
#endif
        std::size_t num_parcels_;  ///< The number of parcels processed by this message
        std::size_t raw_bytes_;    ///< number of bytes processed for the action in
                                   ///< this parcel (uncompressed)

        boost::int64_t buffer_allocate_time_; ///< The time spent for allocating buffers
    };
}}}

#endif // HPX_76311D67_43DA_4B3A_8A2A_14B8A1A266D9

