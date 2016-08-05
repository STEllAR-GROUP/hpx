////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E)
#define HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <boost/cstdint.hpp>

#include <mutex>

namespace hpx { namespace performance_counters { namespace parcels
{
    namespace detail
    {
        /// \brief Collect statistics information about parcels sent and received.
        template <typename Mutex>
        class gatherer
        {
            typedef Mutex mutex_type;

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

        template <typename Mutex>
        inline void gatherer<Mutex>::add_data(data_point const& x)
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

        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::num_parcels(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(num_parcels_, reset);
        }

        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::num_messages(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(num_messages_, reset);
        }

        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::total_time(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(overall_time_, reset);
        }

        template <typename Mutex>
        inline boost::int64_t
        gatherer<Mutex>::total_serialization_time(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(serialization_time_, reset);
        }

#if defined(HPX_HAVE_SECURITY)
        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::total_security_time(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(security_time_, reset);
        }
#endif

        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::total_bytes(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(overall_bytes_, reset);
        }

        template <typename Mutex>
        inline boost::int64_t gatherer<Mutex>::total_raw_bytes(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(overall_raw_bytes_, reset);
        }

        template <typename Mutex>
        inline boost::int64_t
        gatherer<Mutex>::total_buffer_allocate_time(bool reset)
        {
            std::lock_guard<mutex_type> l(acc_mtx);
            return util::get_and_reset_value(buffer_allocate_time_, reset);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    using gatherer = detail::gatherer<lcos::local::spinlock>;
    using gatherer_nolock = detail::gatherer<lcos::local::no_mutex>;
}}}

#endif // HPX_05A1C29B_DB73_463A_8C9D_B8EDC3B69F5E

