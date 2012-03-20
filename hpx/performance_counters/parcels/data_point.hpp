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
          : bytes_(0), timer_(0), num_parcels_(0)
        {}

        std::size_t bytes_;       ///< number of bytes processed for this parcel
        boost::int64_t timer_;    ///< during processing holds start timestamp
                                  ///< after processing holds elapsed time
        std::size_t num_parcels_; ///< The number of parcels precessed by this message
    };
}}}

#endif // HPX_76311D67_43DA_4B3A_8A2A_14B8A1A266D9

