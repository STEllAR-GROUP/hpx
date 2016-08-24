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

#include <cstdint>
#include <mutex>

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
            overall_raw_bytes_(0),
            buffer_allocate_time_(0),
            acc_mtx()
        {}

        void add_data(data_point const& x);

        std::int64_t num_parcels(bool reset);
        std::int64_t num_messages(bool reset);
        std::int64_t total_bytes(bool reset);
        std::int64_t total_raw_bytes(bool reset);
        std::int64_t total_time(bool reset);
        std::int64_t total_serialization_time(bool reset);
#if defined(HPX_HAVE_SECURITY)
        std::int64_t total_security_time(bool reset);
#endif
        std::int64_t total_buffer_allocate_time(bool reset);

    private:
        std::int64_t overall_bytes_;
        std::int64_t overall_time_;
        std::int64_t serialization_time_;
#if defined(HPX_HAVE_SECURITY)
        std::int64_t security_time_;
#endif
        std::int64_t num_parcels_;
        std::int64_t num_messages_;
        std::int64_t overall_raw_bytes_;

        std::int64_t buffer_allocate_time_;

        // Create mutex for accumulator functions.
        mutable mutex_type acc_mtx;
    };

    inline void gatherer::add_data(data_point const& x)
    {
        std::lock_guard<mutex_type> l(acc_mtx);

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

    inline std::int64_t gatherer::num_parcels(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(num_parcels_, reset);
    }

    inline std::int64_t gatherer::num_messages(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(num_messages_, reset);
    }

    inline std::int64_t gatherer::total_time(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_time_, reset);
    }

    inline std::int64_t gatherer::total_serialization_time(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(serialization_time_, reset);
    }

#if defined(HPX_HAVE_SECURITY)
    inline std::int64_t gatherer::total_security_time(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(security_time_, reset);
    }
#endif

    inline std::int64_t gatherer::total_bytes(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_bytes_, reset);
    }

    inline std::int64_t gatherer::total_raw_bytes(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(overall_raw_bytes_, reset);
    }

    inline std::int64_t gatherer::total_buffer_allocate_time(bool reset)
    {
        std::lock_guard<mutex_type> l(acc_mtx);
        return util::get_and_reset_value(buffer_allocate_time_, reset);
    }
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

