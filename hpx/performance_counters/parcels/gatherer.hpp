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
            overall_type_bytes_(0)
        {}

        void add_data(data_point const& x);

        std::size_t num_parcels() const;
        std::size_t num_messages() const;
        std::size_t total_bytes() const;
        std::size_t total_type_bytes() const;
        boost::int64_t total_time() const;
        boost::int64_t total_serialization_time() const;

    private:
        std::size_t overall_bytes_;
        boost::int64_t overall_time_;
        boost::int64_t serialization_time_;
        std::size_t num_parcels_;
        std::size_t num_messages_;
        std::size_t overall_type_bytes_;

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
        overall_type_bytes_ += x.type_bytes_;
        ++num_messages_;
    }

    inline std::size_t gatherer::num_parcels() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return num_parcels_;
    }

    inline std::size_t gatherer::num_messages() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return num_messages_;
    }

    inline boost::int64_t gatherer::total_time() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return overall_time_;
    }

    inline boost::int64_t gatherer::total_serialization_time() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return serialization_time_;
    }

    inline std::size_t gatherer::total_bytes() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return overall_bytes_;
    }

    inline std::size_t gatherer::total_type_bytes() const
    {
        mutex_type::scoped_lock mtx(acc_mtx);
        return overall_type_bytes_;
    }
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

