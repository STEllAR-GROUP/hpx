//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/threads/detail/thread_pool.hpp>

#include <hpx/compat/barrier.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/error_code.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/thread_specific_ptr.hpp>

#include <boost/atomic.hpp>
#include <boost/system/system_error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <mutex>
#include <numeric>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    void thread_pool::init_tss(std::size_t num)
    {
        thread_num_tss_.init_tss(num);
    }

    void thread_pool::deinit_tss()
    {
        thread_num_tss_.deinit_tss();
    }

    ///////////////////////////////////////////////////////////////////////////

    thread_pool::thread_pool(
            threads::policies::callback_notifier& notifier,
            char const* pool_name, policies::scheduler_mode m)
            : notifier_(notifier),
              pool_name_(pool_name),
              thread_count_(0),
              used_processing_units_(),
              mode_(m)
    {
        timestamp_scale_ = 1.0;
    }

    thread_pool::~thread_pool(){}

    ///////////////////////////////////////////////////////////////////////////

    mask_cref_type thread_pool::get_used_processing_units() const
    {
        return used_processing_units_;
    }

    ///////////////////////////////////////////////////////////////////////////

    thread_state thread_pool::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state(id, new_state, //-V107
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t thread_pool::get_worker_thread_num() const
    {
        return thread_num_tss_.get_worker_thread_num();
    }

    std::int64_t thread_pool::get_scheduler_utilization() const
    {
        return (std::accumulate(tasks_active_.begin(), tasks_active_.end(),
            std::int64_t(0)) * 100) / thread_count_.load();
    }


    ///////////////////////////////////////////////////////////////////////////
    void thread_pool::stop (
        std::unique_lock<compat::mutex>& l, bool blocking)
    {
        HPX_ASSERT(l.owns_lock());

        return stop_locked(l, blocking);
    }


    ///////////////////////////////////////////////////////////////////////////
    // performance counters
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
    std::int64_t thread_pool::
    get_executed_threads(std::size_t num, bool reset)
    {
        std::int64_t executed_threads = 0;
        std::int64_t reset_executed_threads = 0;

        if (num != std::size_t(-1))
        {
            executed_threads = executed_threads_[num];
            reset_executed_threads = reset_executed_threads_[num];

            if (reset)
                reset_executed_threads_[num] = executed_threads;
        }
        else
        {
            executed_threads = std::accumulate(executed_threads_.begin(),
                                               executed_threads_.end(), std::int64_t(0));
            reset_executed_threads = std::accumulate(
                    reset_executed_threads_.begin(),
                    reset_executed_threads_.end(), std::int64_t(0));

            if (reset)
            {
                std::copy(executed_threads_.begin(), executed_threads_.end(),
                          reset_executed_threads_.begin());
            }
        }

        HPX_ASSERT(executed_threads >= reset_executed_threads);

        return executed_threads - reset_executed_threads;
    }

    std::int64_t thread_pool::
        get_executed_thread_phases(std::size_t num, bool reset)
    {
        std::int64_t executed_phases = 0;
        std::int64_t reset_executed_phases = 0;

        if (num != std::size_t(-1))
        {
            executed_phases = executed_thread_phases_[num];
            reset_executed_phases = reset_executed_thread_phases_[num];

            if (reset)
                reset_executed_thread_phases_[num] = executed_phases;
        }
        else
        {
            executed_phases = std::accumulate(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), std::int64_t(0));
            reset_executed_phases = std::accumulate(
                reset_executed_thread_phases_.begin(),
                reset_executed_thread_phases_.end(), std::int64_t(0));

            if (reset)
            {
                std::copy(executed_thread_phases_.begin(),
                    executed_thread_phases_.end(),
                    reset_executed_thread_phases_.begin());
            }
        }

        HPX_ASSERT(executed_phases >= reset_executed_phases);

        return executed_phases - reset_executed_phases;
    }

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
    std::int64_t thread_pool::
        get_thread_phase_duration(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0ul;
        std::int64_t num_phases = 0l;
        std::uint64_t reset_exec_total = 0ul;
        std::int64_t reset_num_phases = 0l;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            num_phases = executed_thread_phases_[num];

            reset_exec_total = reset_thread_phase_duration_times_[num];
            reset_num_phases = reset_thread_phase_duration_[num];

            if (reset)
            {
                reset_thread_phase_duration_[num] = num_phases;
                reset_thread_phase_duration_times_[num] = exec_total;
            }
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            num_phases = std::accumulate(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), std::int64_t(0));

            reset_exec_total = std::accumulate(
                reset_thread_phase_duration_times_.begin(),
                reset_thread_phase_duration_times_.end(), std::uint64_t(0));
            reset_num_phases = std::accumulate(
                reset_thread_phase_duration_.begin(),
                reset_thread_phase_duration_.end(), std::int64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_thread_phase_duration_times_.begin());
                std::copy(executed_thread_phases_.begin(),
                    executed_thread_phases_.end(),
                    reset_thread_phase_duration_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(num_phases >= reset_num_phases);

        exec_total -= reset_exec_total;
        num_phases -= reset_num_phases;

        return std::uint64_t(
                (double(exec_total) * timestamp_scale_) / double(num_phases)
            );
    }

    std::int64_t thread_pool::
        get_thread_duration(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0ul;
        std::int64_t num_threads = 0l;
        std::uint64_t reset_exec_total = 0ul;
        std::int64_t reset_num_threads = 0l;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            num_threads = executed_threads_[num];

            reset_exec_total = reset_thread_duration_times_[num];
            reset_num_threads = reset_thread_duration_[num];

            if (reset)
            {
                reset_thread_duration_[num] = num_threads;
                reset_thread_duration_times_[num] = exec_total;
            }
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            num_threads = std::accumulate(executed_threads_.begin(),
                executed_threads_.end(), std::int64_t(0));

            reset_exec_total = std::accumulate(
                reset_thread_duration_times_.begin(),
                reset_thread_duration_times_.end(),
                std::uint64_t(0));
            reset_num_threads = std::accumulate(
                reset_thread_duration_.begin(),
                reset_thread_duration_.end(),
                std::int64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_thread_duration_times_.begin());
                std::copy(executed_threads_.begin(),
                    executed_threads_.end(),
                    reset_thread_duration_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(num_threads >= reset_num_threads);

        exec_total -= reset_exec_total;
        num_threads -= reset_num_threads;

        return std::uint64_t(
                (double(exec_total) * timestamp_scale_) / double(num_threads)
            );
    }

    std::int64_t thread_pool::
        get_thread_phase_overhead(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0;
        std::uint64_t tfunc_total = 0;
        std::int64_t num_phases = 0;

        std::uint64_t reset_exec_total = 0;
        std::uint64_t reset_tfunc_total = 0;
        std::int64_t reset_num_phases = 0;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            tfunc_total = tfunc_times_[num];
            num_phases = executed_thread_phases_[num];

            reset_exec_total =  reset_thread_phase_overhead_times_[num];
            reset_tfunc_total = reset_thread_phase_overhead_times_total_[num];
            reset_num_phases =  reset_thread_phase_overhead_[num];

            if (reset)
            {
                reset_thread_phase_overhead_times_[num] = exec_total;
                reset_thread_phase_overhead_times_total_[num] = tfunc_total;
                reset_thread_phase_overhead_[num] = num_phases;
            }
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            tfunc_total = std::accumulate(tfunc_times_.begin(),
                tfunc_times_.end(), std::uint64_t(0));
            num_phases = std::accumulate(
                executed_thread_phases_.begin(),
                executed_thread_phases_.end(), std::int64_t(0));

            reset_exec_total = std::accumulate(
                reset_thread_phase_overhead_times_.begin(),
                reset_thread_phase_overhead_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(
                reset_thread_phase_overhead_times_total_.begin(),
                reset_thread_phase_overhead_times_total_.end(),
                std::uint64_t(0));
            reset_num_phases = std::accumulate(
                reset_thread_phase_overhead_.begin(),
                reset_thread_phase_overhead_.end(), std::int64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_thread_phase_overhead_times_.begin());
                std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                    reset_thread_phase_overhead_times_total_.begin());
                std::copy(executed_thread_phases_.begin(),
                    executed_thread_phases_.end(),
                    reset_thread_phase_overhead_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);
        HPX_ASSERT(num_phases >= reset_num_phases);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;
        num_phases -= reset_num_phases;

        if (num_phases == 0)        // avoid division by zero
            return 0;

        HPX_ASSERT(tfunc_total >= exec_total);

        return std::uint64_t(
                double((tfunc_total - exec_total) * timestamp_scale_) /
                double(num_phases)
            );
    }

    std::int64_t thread_pool::
        get_thread_overhead(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0;
        std::uint64_t tfunc_total = 0;
        std::int64_t num_threads = 0;

        std::uint64_t reset_exec_total = 0;
        std::uint64_t reset_tfunc_total = 0;
        std::int64_t reset_num_threads = 0;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            tfunc_total = tfunc_times_[num];
            num_threads = executed_threads_[num];

            reset_exec_total =  reset_thread_overhead_times_[num];
            reset_tfunc_total = reset_thread_overhead_times_total_[num];
            reset_num_threads =  reset_thread_overhead_[num];

            if (reset)
            {
                reset_thread_overhead_times_[num] = exec_total;
                reset_thread_overhead_times_total_[num] = tfunc_total;
                reset_thread_overhead_[num] = num_threads;
            }
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            tfunc_total = std::accumulate(tfunc_times_.begin(),
                tfunc_times_.end(), std::uint64_t(0));
            num_threads = std::accumulate(executed_threads_.begin(),
                executed_threads_.end(), std::int64_t(0));

            reset_exec_total = std::accumulate(
                reset_thread_overhead_times_.begin(),
                reset_thread_overhead_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(
                reset_thread_overhead_times_total_.begin(),
                reset_thread_overhead_times_total_.end(),
                std::uint64_t(0));
            reset_num_threads = std::accumulate(
                reset_thread_overhead_.begin(),
                reset_thread_overhead_.end(), std::int64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_thread_overhead_times_.begin());
                std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                    reset_thread_overhead_times_total_.begin());
                std::copy(executed_threads_.begin(),
                    executed_threads_.end(),
                    reset_thread_overhead_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);
        HPX_ASSERT(num_threads >= reset_num_threads);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;
        num_threads -= reset_num_threads;

        if (num_threads == 0)        // avoid division by zero
            return 0;

        HPX_ASSERT(tfunc_total >= exec_total);

        return std::uint64_t(
                double((tfunc_total - exec_total) * timestamp_scale_) /
                double(num_threads)
            );
    }

    std::int64_t thread_pool::
        get_cumulative_thread_duration(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0ul;
        std::uint64_t reset_exec_total = 0ul;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            reset_exec_total = reset_cumulative_thread_duration_[num];

            if (reset)
                reset_cumulative_thread_duration_[num] = exec_total;
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            reset_exec_total = std::accumulate(
                reset_cumulative_thread_duration_.begin(),
                reset_cumulative_thread_duration_.end(),
                std::uint64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_cumulative_thread_duration_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);

        exec_total -= reset_exec_total;

        return std::uint64_t(double(exec_total) * timestamp_scale_);
    }

    std::int64_t thread_pool::
        get_cumulative_thread_overhead(std::size_t num, bool reset)
    {
        std::uint64_t exec_total = 0ul;
        std::uint64_t reset_exec_total = 0ul;
        std::uint64_t tfunc_total = 0ul;
        std::uint64_t reset_tfunc_total = 0ul;

        if (num != std::size_t(-1))
        {
            exec_total = exec_times_[num];
            tfunc_total = tfunc_times_[num];

            reset_exec_total = reset_cumulative_thread_overhead_[num];
            reset_tfunc_total = reset_cumulative_thread_overhead_total_[num];

            if (reset)
            {
                reset_cumulative_thread_overhead_[num] = exec_total;
                reset_cumulative_thread_overhead_total_[num] = tfunc_total;
            }
        }
        else
        {
            exec_total = std::accumulate(exec_times_.begin(),
                exec_times_.end(), std::uint64_t(0));
            reset_exec_total = std::accumulate(
                reset_cumulative_thread_overhead_.begin(),
                reset_cumulative_thread_overhead_.end(),
                std::uint64_t(0));

            tfunc_total = std::accumulate(tfunc_times_.begin(),
                tfunc_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(
                reset_cumulative_thread_overhead_total_.begin(),
                reset_cumulative_thread_overhead_total_.end(),
                std::uint64_t(0));

            if (reset)
            {
                std::copy(exec_times_.begin(), exec_times_.end(),
                    reset_cumulative_thread_overhead_.begin());
                std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                    reset_cumulative_thread_overhead_total_.begin());
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        return std::uint64_t(
                (double(tfunc_total) - double(exec_total)) * timestamp_scale_
            );
    }
#endif
#endif

    std::int64_t thread_pool::get_cumulative_duration(std::size_t num, bool reset)
    {
        std::uint64_t tfunc_total = 0ul;
        std::uint64_t reset_tfunc_total = 0ul;

        if (num != std::size_t(-1))
        {
            tfunc_total = tfunc_times_[num];
            reset_tfunc_total = reset_tfunc_times_[num];

            if (reset)
                reset_tfunc_times_[num] = tfunc_total;
        }
        else
        {
            tfunc_total = std::accumulate(tfunc_times_.begin(),
                tfunc_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(
                reset_tfunc_times_.begin(), reset_tfunc_times_.end(),
                std::uint64_t(0));

            if (reset)
            {
                std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                    reset_tfunc_times_.begin());
            }
        }

        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        tfunc_total -= reset_tfunc_total;

        return std::uint64_t(double(tfunc_total) * timestamp_scale_);
    }

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
    ///////////////////////////////////////////////////////////////////////////
    std::int64_t thread_pool::avg_idle_rate(bool reset)
    {
        std::uint64_t exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total = std::accumulate(
            reset_idle_rate_time_.begin(),
            reset_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total = std::accumulate(
            reset_idle_rate_time_total_.begin(),
            reset_idle_rate_time_total_.end(), std::uint64_t(0));

        if (reset)
        {
            std::copy(exec_times_.begin(), exec_times_.end(),
                reset_idle_rate_time_.begin());
            std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                reset_idle_rate_time_total_.begin());
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == 0)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total >= exec_total);

        double const percent = 1. - (double(exec_total) / double(tfunc_total));
        return std::int64_t(10000. * percent);   // 0.01 percent
    }

    std::int64_t thread_pool::avg_idle_rate(
        std::size_t num_thread, bool reset)
    {
        std::uint64_t exec_time = exec_times_[num_thread];
        std::uint64_t tfunc_time = tfunc_times_[num_thread];
        std::uint64_t reset_exec_time = reset_idle_rate_time_[num_thread];
        std::uint64_t reset_tfunc_time = reset_idle_rate_time_total_[num_thread];

        if (reset)
        {
            reset_idle_rate_time_[num_thread] = exec_time;
            reset_idle_rate_time_total_[num_thread] = tfunc_time;
        }

        HPX_ASSERT(exec_time >= reset_exec_time);
        HPX_ASSERT(tfunc_time >= reset_tfunc_time);

        exec_time -= reset_exec_time;
        tfunc_time -= reset_tfunc_time;

        if (tfunc_time == 0)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_time > exec_time);

        double const percent = 1. - (double(exec_time) / double(tfunc_time));
        return std::int64_t(10000. * percent);   // 0.01 percent
    }
#endif

    std::int64_t thread_pool::get_idle_loop_count(std::size_t num) const
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(idle_loop_counts_.begin(),
                idle_loop_counts_.end(), 0ll);
        }
        return idle_loop_counts_[num];
    }

    std::int64_t thread_pool::get_busy_loop_count(std::size_t num) const
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(busy_loop_counts_.begin(),
                busy_loop_counts_.end(), 0ll);
        }
        return busy_loop_counts_[num];
    }
}}}

