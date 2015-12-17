////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E)
#define HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/cstdint.hpp>
#include <boost/thread/locks.hpp>

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
#if defined(HPX_HAVE_SECURITY)
            security_time_(0),
#endif
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
#if defined(HPX_HAVE_SECURITY)
        boost::int64_t total_security_time(bool reset);
#endif
        boost::int64_t total_buffer_allocate_time(bool reset);

    private:
        boost::int64_t overall_bytes_;
        boost::int64_t overall_time_;
        boost::int64_t serialization_time_;
#if defined(HPX_HAVE_SECURITY)
        boost::int64_t security_time_;
#endif
        boost::int64_t num_parcels_;
        boost::int64_t num_messages_;
        boost::int64_t overall_raw_bytes_;

        boost::int64_t buffer_allocate_time_;

        // Create mutex for accumulator functions.
        mutable mutex_type acc_mtx;
    };

    inline void gatherer::add_data(data_point const& x)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);

        overall_bytes_ += x.bytes_;
        overall_time_ += x.time_;
        serialization_time_ += x.serialization_time_;
#if defined(HPX_HAVE_SECURITY)
        security_time_ += x.security_time_;
#endif
        num_parcels_ += x.num_parcels_;
        overall_raw_bytes_ += x.raw_bytes_;
        ++num_messages_;
        buffer_allocate_time_ += x.buffer_allocate_time_;
    }

    inline boost::int64_t gatherer::num_parcels(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(num_parcels_, reset);
    }

    inline boost::int64_t gatherer::num_messages(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(num_messages_, reset);
    }

    inline boost::int64_t gatherer::total_time(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_time_, reset);
    }

    inline boost::int64_t gatherer::total_serialization_time(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(serialization_time_, reset);
    }

#if defined(HPX_HAVE_SECURITY)
    inline boost::int64_t gatherer::total_security_time(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(security_time_, reset);
    }
#endif

    inline boost::int64_t gatherer::total_bytes(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_bytes_, reset);
    }

    inline boost::int64_t gatherer::total_raw_bytes(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_raw_bytes_, reset);
    }

    inline boost::int64_t gatherer::total_buffer_allocate_time(bool reset)
    {
        boost::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(buffer_allocate_time_, reset);
    }
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

