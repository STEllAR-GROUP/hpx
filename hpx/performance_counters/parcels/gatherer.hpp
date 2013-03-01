////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E)
#define HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/get_and_reset_value.hpp>

namespace hpx { namespace performance_counters { namespace parcels
{
    /// \brief Collect statistics information about parcels sent and received.
    class gatherer
    {
        typedef hpx::util::spinlock mutex_type;

    public:
        gatherer()
          : overall_bytes_(0),
            overall_time_(0),
            serialization_time_(0),
            num_parcels_(0),
            num_messages_(0),
            overall_raw_bytes_(0)
        {}

        void add_data(data_point const& x);

        boost::int64_t num_parcels(bool reset);
        boost::int64_t num_messages(bool reset);
        boost::int64_t total_bytes(bool reset);
        boost::int64_t total_raw_bytes(bool reset);
        boost::int64_t total_time(bool reset);
        boost::int64_t total_serialization_time(bool reset);

    private:
        boost::int64_t overall_bytes_;
        boost::int64_t overall_time_;
        boost::int64_t serialization_time_;
        boost::int64_t num_parcels_;
        boost::int64_t num_messages_;
        boost::int64_t overall_raw_bytes_;

        // Create mutex for accumulator functions.
        mutable mutex_type acc_mtx;
    };

    inline void gatherer::add_data(data_point const& x)
    {
        mutex_type::scoped_lock mtx(acc_mtx);

        overall_bytes_ += x.bytes_;
        overall_time_ += x.time_;
        serialization_time_ += x.serialization_time_;
        num_parcels_ += x.num_parcels_;
        overall_raw_bytes_ += x.raw_bytes_;
        ++num_messages_;
    }

    inline boost::int64_t gatherer::num_parcels(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(num_parcels_, reset);
    }

    inline boost::int64_t gatherer::num_messages(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(num_messages_, reset);
    }

    inline boost::int64_t gatherer::total_time(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(overall_time_, reset);
    }

    inline boost::int64_t gatherer::total_serialization_time(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(serialization_time_, reset);
    }

    inline boost::int64_t gatherer::total_bytes(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(overall_bytes_, reset);
    }

    inline boost::int64_t gatherer::total_raw_bytes(bool reset)
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return util::get_and_reset_value(overall_raw_bytes_, reset);
    }
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

