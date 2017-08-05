//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SCHEDULED_THREAD_POOL_IMPL_HPP)
#define HPX_SCHEDULED_THREAD_POOL_IMPL_HPP

#include <hpx/compat/barrier.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/compat/thread.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_info.hpp>
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <algorithm>
#include <numeric>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper
    {
        init_tss_helper(
            scheduled_thread_pool<Scheduler>& pool,
                std::size_t pool_thread_num, std::size_t offset)
          : pool_(pool)
          , thread_num_(pool_thread_num)
        {
            pool.notifier_.on_start_thread(pool_thread_num);
            threads::get_thread_manager().init_tss(pool_thread_num + offset);
            pool.sched_->Scheduler::on_start_thread(pool_thread_num);
        }
        ~init_tss_helper()
        {
            pool_.sched_->Scheduler::on_stop_thread(thread_num_);
            threads::get_thread_manager().deinit_tss();
            pool_.notifier_.on_stop_thread(thread_num_);
        }

        scheduled_thread_pool<Scheduler>& pool_;
        std::size_t thread_num_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    scheduled_thread_pool<Scheduler>::scheduled_thread_pool(
            std::unique_ptr<Scheduler> sched,
            threads::policies::callback_notifier& notifier, std::size_t index,
            char const* pool_name, policies::scheduler_mode m,
            std::size_t thread_offset)
        : thread_pool_base(notifier, index, pool_name, m, thread_offset)
        , sched_(std::move(sched))
        , thread_count_(0)
    {
        sched_->set_parent_pool(this);
    }

    template <typename Scheduler>
    scheduled_thread_pool<Scheduler>::~scheduled_thread_pool()
    {
        if (!threads_.empty())
        {
            if (!sched_->Scheduler::has_reached_state(state_suspended))
            {
                // still running
                lcos::local::no_mutex mtx;
                std::unique_lock<lcos::local::no_mutex> l(mtx);
                stop_locked(l);
            }
            threads_.clear();
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::print_pool(std::ostream& os)
    {
        os << "[pool \"" << id_.name_ << "\", #" << id_.index_
           << "] with scheduler " << sched_->Scheduler::get_scheduler_name()
           << "\n"
           << "is running on PUs : \n";
        os << std::hex << HPX_CPU_MASK_PREFIX << used_processing_units_ << '\n';
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::report_error(
        std::size_t num, std::exception_ptr const& e)
    {
        sched_->Scheduler::set_all_states(state_terminating);
        this->thread_pool_base::report_error(num, e);
        sched_->Scheduler::on_error(num, e);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    hpx::state scheduled_thread_pool<Scheduler>::get_state() const
    {
        // get_worker_thread_num returns the global thread number which
        // might be too large. This function might get called from within
        // background_work inside the os executors
        if (thread_count_ != 0)
        {
            std::size_t num_thread = get_worker_thread_num() % thread_count_;
            if (num_thread != std::size_t(-1))
                return get_state(num_thread);
        }
        return sched_->Scheduler::get_minmax_state().second;
    }

    template <typename Scheduler>
    hpx::state scheduled_thread_pool<Scheduler>::get_state(
        std::size_t num_thread) const
    {
        HPX_ASSERT(num_thread != std::size_t(-1));
        return sched_->Scheduler::get_state(num_thread).load();
    }

    template <typename Scheduler>
    template <typename Lock>
    void scheduled_thread_pool<Scheduler>::stop_locked(Lock& l, bool blocking)
    {
        LTM_(info) << "stop: " << id_.name_ << " blocking(" << std::boolalpha
                   << blocking << ")";

        if (!threads_.empty())
        {
            // set state to stopping
            sched_->Scheduler::set_all_states(state_stopping);

            // make sure we're not waiting
            sched_->Scheduler::do_some_work(std::size_t(-1));

            if (blocking)
            {
                for (std::size_t i = 0; i != threads_.size(); ++i)
                {
                    // make sure no OS thread is waiting
                    LTM_(info) << "stop: " << id_.name_ << " notify_all";

                    sched_->Scheduler::do_some_work(std::size_t(-1));

                    LTM_(info)                                        //-V128
                        << "stop: " << id_.name_ << " join:" << i;    //-V128

                    // unlock the lock while joining
                    util::unlock_guard<Lock> ul(l);
                    threads_[i].join();
                }
                threads_.clear();
            }
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::stop(
        std::unique_lock<compat::mutex>& l, bool blocking)
    {
        HPX_ASSERT(l.owns_lock());
        return stop_locked(l, blocking);
    }

    template <typename Scheduler>
    bool hpx::threads::detail::scheduled_thread_pool<Scheduler>::run(
        std::unique_lock<compat::mutex>& l, compat::barrier& startup,
        std::size_t pool_threads)
    {
        HPX_ASSERT(l.owns_lock());

        LTM_(info)    //-V128
            << "run: " << id_.name_
            << " number of processing units available: "    //-V128
            << threads::hardware_concurrency();
        LTM_(info)    //-V128
            << "run: " << id_.name_ << " creating " << pool_threads
            << " OS thread(s)";    //-V128

        if (0 == pool_threads)
        {
            HPX_THROW_EXCEPTION(
                bad_parameter, "run", "number of threads is zero");
        }

        if (!threads_.empty() ||
            sched_->Scheduler::has_reached_state(state_running))
        {
            return true;    // do nothing if already running
        }

        init_perf_counter_data(pool_threads);
        this->init_pool_time_scale();

        LTM_(info) << "run: " << id_.name_
                   << " timestamp_scale: " << timestamp_scale_;    //-V128

        // run threads and wait for initialization to complete
        std::size_t thread_num = 0;
        try
        {
            auto const& rp = get_resource_partitioner();
            topology const& topology_ = get_topology();

            for (/**/; thread_num != pool_threads; ++thread_num)
            {
                std::size_t global_thread_num = this->thread_offset_ + thread_num;
                threads::mask_cref_type mask = rp.get_pu_mask(global_thread_num);

                // thread_num ordering: 1. threads of default pool
                //                      2. threads of first special pool
                //                      3. etc.
                // get_pu_mask expects index according to ordering of masks
                // in affinity_data::affinity_masks_
                // which is in order of occupied PU
                LTM_(info)    //-V128
                    << "run: " << id_.name_ << " create OS thread "
                    << global_thread_num    //-V128
                    << ": will run on processing units within this mask: "
                    << std::hex << HPX_CPU_MASK_PREFIX << mask;

                // create a new thread
                threads_.push_back(
                    compat::thread(&scheduled_thread_pool::thread_func, this,
                        thread_num, std::ref(topology_), std::ref(startup)));

                // set the new threads affinity (on Windows systems)
                if (!any(mask))
                {
                    LTM_(debug)    //-V128
                        << "run: " << id_.name_
                        << " setting thread affinity on OS thread "    //-V128
                        << global_thread_num << " was explicitly disabled.";
                }
            }
        }
        catch (std::exception const& e)
        {
            LTM_(always) << "run: " << id_.name_
                         << " failed with: " << e.what();

            // trigger the barrier
            pool_threads -= thread_num;
            while (pool_threads-- != 0)
                startup.wait();

            stop_locked(l);
            threads_.clear();

            return false;
        }

        LTM_(info) << "run: " << id_.name_ << " running";
        return true;
    }

    template <typename Scheduler>
    void hpx::threads::detail::scheduled_thread_pool<Scheduler>::thread_func(
        std::size_t thread_num, topology const& topology,
        compat::barrier& startup)
    {
        std::size_t global_thread_num = thread_num + this->thread_offset_;

        // Set the affinity for the current thread.
        threads::mask_cref_type mask =
            get_resource_partitioner().get_pu_mask(global_thread_num);

        if (LHPX_ENABLED(debug))
            topology.write_to_log();

        error_code ec(lightweight);
        if (any(mask))
        {
            topology.set_thread_affinity_mask(mask, ec);
            if (ec)
            {
                LTM_(warning)    //-V128
                    << "thread_func: " << id_.name_
                    << " setting thread affinity on OS thread "    //-V128
                    << global_thread_num
                    << " failed with: " << ec.get_message();
            }
        }
        else
        {
            LTM_(debug)    //-V128
                << "thread_func: " << id_.name_
                << " setting thread affinity on OS thread "    //-V128
                << global_thread_num << " was explicitly disabled.";
        }

        // Setting priority of worker threads to a lower priority, this
        // needs to
        // be done in order to give the parcel pool threads higher
        // priority
        if ((mode_ & policies::reduce_thread_priority) &&
            any(mask & used_processing_units_))
        {
            topology.reduce_thread_priority(ec);
            if (ec)
            {
                LTM_(warning)    //-V128
                    << "thread_func: " << id_.name_
                    << " reducing thread priority on OS thread "    //-V128
                    << global_thread_num
                    << " failed with: " << ec.get_message();
            }
        }

        // manage the number of this thread in its TSS
        init_tss_helper<Scheduler> tss_helper(
            *this, thread_num, this->thread_offset_);

        // wait for all threads to start up before before starting HPX work
        startup.wait();

        {
            LTM_(info)    //-V128
                << "thread_func: " << id_.name_
                << " starting OS thread: " << thread_num;    //-V128

            try
            {
                try
                {
                    manage_active_thread_count count(thread_count_);

                    // run the work queue
                    hpx::threads::coroutines::prepare_main_thread main_thread;

                    // run main Scheduler loop until terminated
                    detail::scheduling_counters counters(
                        executed_threads_[thread_num],
                        executed_thread_phases_[thread_num],
                        tfunc_times_[thread_num],
                        exec_times_[thread_num],
                        idle_loop_counts_[thread_num],
                        busy_loop_counts_[thread_num],
                        tasks_active_[thread_num]);

                    detail::scheduling_callbacks callbacks(
                        util::bind(    //-V107
                            &policies::scheduler_base::idle_callback,
                            std::ref(sched_), global_thread_num),
                        detail::scheduling_callbacks::callback_type());

                    if (mode_ & policies::do_background_work)
                    {
                        callbacks.background_ = util::bind(    //-V107
                            &policies::scheduler_base::background_callback,
                            std::ref(sched_), global_thread_num);
                    }

                    sched_->Scheduler::set_scheduler_mode(mode_);
                    detail::scheduling_loop(
                        thread_num, *sched_, counters, callbacks);

                    // the OS thread is allowed to exit only if no more HPX
                    // threads exist or if some other thread has terminated
                    HPX_ASSERT(
                        !sched_->Scheduler::get_thread_count(unknown,
                                   thread_priority_default, thread_num) ||
                        sched_->Scheduler::get_state(thread_num) ==
                            state_terminating);
                }
                catch (hpx::exception const& e)
                {
                    LFATAL_    //-V128
                        << "thread_func: " << id_.name_
                        << " thread_num:" << global_thread_num    //-V128
                        << " : caught hpx::exception: " << e.what()
                        << ", aborted thread execution";

                    report_error(global_thread_num, std::current_exception());
                    return;
                }
                catch (boost::system::system_error const& e)
                {
                    LFATAL_    //-V128
                        << "thread_func: " << id_.name_
                        << " thread_num:" << global_thread_num    //-V128
                        << " : caught boost::system::system_error: " << e.what()
                        << ", aborted thread execution";

                    report_error(global_thread_num, std::current_exception());
                    return;
                }
                catch (std::exception const& e)
                {
                    // Repackage exceptions to avoid slicing.
                    hpx::throw_with_info(
                        hpx::exception(unhandled_exception, e.what()));
                }
            }
            catch (...)
            {
                LFATAL_    //-V128
                    << "thread_func: " << id_.name_
                    << " thread_num:" << global_thread_num    //-V128
                    << " : caught unexpected "                //-V128
                       "exception, aborted thread execution";

                report_error(global_thread_num, std::current_exception());
                return;
            }

            LTM_(info)    //-V128
                << "thread_func: " << id_.name_
                << " thread_num: " << global_thread_num
                << " : ending OS thread, "    //-V128
                   "executed "
                << executed_threads_[global_thread_num] << " HPX threads";
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::create_thread(
        thread_init_data& data, thread_id_type& id,
        thread_state_enum initial_state, bool run_now, error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_->Scheduler::is_state(state_running))
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "thread_pool<Scheduler>::create_thread",
                "invalid state: thread pool is not running");
            return;
        }

        detail::create_thread(
            sched_.get(), data, id, initial_state, run_now, ec);    //-V601
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::create_work(
        thread_init_data& data, thread_state_enum initial_state, error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_->Scheduler::is_state(state_running))
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "thread_pool<Scheduler>::create_work",
                "invalid state: thread pool is not running");
            return;
        }

        detail::create_work(sched_.get(), data, initial_state, ec);    //-V601
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_state scheduled_thread_pool<Scheduler>::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state(id, new_state, //-V107
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    template <typename Scheduler>
    thread_id_type scheduled_thread_pool<Scheduler>::set_state(
        util::steady_time_point const& abs_time, thread_id_type const& id,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, error_code& ec)
    {
        return detail::set_thread_state_timed(*sched_, abs_time, id, newstate,
            newstate_ex, priority, get_worker_thread_num(), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // performance counters
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_threads(
        std::size_t num, bool reset)
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
            reset_executed_threads =
                std::accumulate(reset_executed_threads_.begin(),
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

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_thread_phases(
        std::size_t num, bool reset)
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
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_phase_duration(
        std::size_t num, bool reset)
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

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_duration(
        std::size_t num, bool reset)
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
            exec_total = std::accumulate(
                exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
            num_threads = std::accumulate(executed_threads_.begin(),
                executed_threads_.end(), std::int64_t(0));

            reset_exec_total =
                std::accumulate(reset_thread_duration_times_.begin(),
                    reset_thread_duration_times_.end(),
                    std::uint64_t(0));
            reset_num_threads = std::accumulate(reset_thread_duration_.begin(),
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
            (double(exec_total) * timestamp_scale_) / double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_phase_overhead(
        std::size_t num, bool reset)
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

            reset_exec_total = reset_thread_phase_overhead_times_[num];
            reset_tfunc_total = reset_thread_phase_overhead_times_total_[num];
            reset_num_phases = reset_thread_phase_overhead_[num];

            if (reset)
            {
                reset_thread_phase_overhead_times_[num] = exec_total;
                reset_thread_phase_overhead_times_total_[num] = tfunc_total;
                reset_thread_phase_overhead_[num] = num_phases;
            }
        }
        else
        {
            exec_total = std::accumulate(
                exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
            tfunc_total = std::accumulate(
                tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
            num_phases = std::accumulate(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), std::int64_t(0));

            reset_exec_total =
                std::accumulate(reset_thread_phase_overhead_times_.begin(),
                    reset_thread_phase_overhead_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(
                reset_thread_phase_overhead_times_total_.begin(),
                reset_thread_phase_overhead_times_total_.end(),
                std::uint64_t(0));
            reset_num_phases =
                std::accumulate(reset_thread_phase_overhead_.begin(),
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

        if (num_phases == 0)    // avoid division by zero
            return 0;

        HPX_ASSERT(tfunc_total >= exec_total);

        return std::uint64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) /
            double(num_phases));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_overhead(
        std::size_t num, bool reset)
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

            reset_exec_total = reset_thread_overhead_times_[num];
            reset_tfunc_total = reset_thread_overhead_times_total_[num];
            reset_num_threads = reset_thread_overhead_[num];

            if (reset)
            {
                reset_thread_overhead_times_[num] = exec_total;
                reset_thread_overhead_times_total_[num] = tfunc_total;
                reset_thread_overhead_[num] = num_threads;
            }
        }
        else
        {
            exec_total = std::accumulate(
                exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
            tfunc_total = std::accumulate(
                tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
            num_threads = std::accumulate(executed_threads_.begin(),
                executed_threads_.end(), std::int64_t(0));

            reset_exec_total =
                std::accumulate(reset_thread_overhead_times_.begin(),
                    reset_thread_overhead_times_.end(), std::uint64_t(0));
            reset_tfunc_total =
                std::accumulate(reset_thread_overhead_times_total_.begin(),
                    reset_thread_overhead_times_total_.end(),
                    std::uint64_t(0));
            reset_num_threads = std::accumulate(reset_thread_overhead_.begin(),
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

        if (num_threads == 0)    // avoid division by zero
            return 0;

        HPX_ASSERT(tfunc_total >= exec_total);

        return std::uint64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) /
            double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_cumulative_thread_duration(
        std::size_t num, bool reset)
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
            exec_total = std::accumulate(
                exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
            reset_exec_total =
                std::accumulate(reset_cumulative_thread_duration_.begin(),
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

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_thread_overhead(
        std::size_t num, bool reset)
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
            exec_total = std::accumulate(
                exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
            reset_exec_total =
                std::accumulate(reset_cumulative_thread_overhead_.begin(),
                    reset_cumulative_thread_overhead_.end(),
                    std::uint64_t(0));

            tfunc_total = std::accumulate(
                tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
            reset_tfunc_total =
                std::accumulate(reset_cumulative_thread_overhead_total_.begin(),
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
            (double(tfunc_total) - double(exec_total)) * timestamp_scale_);
    }
#endif
#endif

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_cumulative_duration(
        std::size_t num, bool reset)
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
            tfunc_total = std::accumulate(
                tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
            reset_tfunc_total = std::accumulate(reset_tfunc_times_.begin(),
                reset_tfunc_times_.end(), std::uint64_t(0));

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
#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_creation_idle_rate(
        std::size_t, bool reset)
    {
        double const creation_total =
            static_cast<double>(sched_->Scheduler::get_creation_time(reset));

        std::uint64_t exec_total = std::accumulate(
            exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(
            tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total =
            std::accumulate(reset_creation_idle_rate_time_.begin(),
                reset_creation_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total =
            std::accumulate(reset_creation_idle_rate_time_total_.begin(),
                reset_creation_idle_rate_time_total_.end(),
                std::uint64_t(0));

        if (reset)
        {
            std::copy(exec_times_.begin(), exec_times_.end(),
                reset_creation_idle_rate_time_.begin());
            std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                reset_creation_idle_rate_time_.begin());
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == exec_total)    // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent =
            (creation_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_cleanup_idle_rate(
        std::size_t, bool reset)
    {
        double const cleanup_total =
            static_cast<double>(sched_->Scheduler::get_cleanup_time(reset));

        std::uint64_t exec_total = std::accumulate(
            exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(
            tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total =
            std::accumulate(reset_cleanup_idle_rate_time_.begin(),
                reset_cleanup_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total =
            std::accumulate(reset_cleanup_idle_rate_time_total_.begin(),
                reset_cleanup_idle_rate_time_total_.end(),
                std::uint64_t(0));

        if (reset)
        {
            std::copy(exec_times_.begin(), exec_times_.end(),
                reset_cleanup_idle_rate_time_.begin());
            std::copy(tfunc_times_.begin(), tfunc_times_.end(),
                reset_cleanup_idle_rate_time_.begin());
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        if (tfunc_total == exec_total)    // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent =
            (cleanup_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }
#endif

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_idle_rate_all(bool reset)
    {
        std::uint64_t exec_total = std::accumulate(
            exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(
            tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total =
            std::accumulate(reset_idle_rate_time_.begin(),
                reset_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total =
            std::accumulate(reset_idle_rate_time_total_.begin(),
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

        if (tfunc_total == 0)    // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total >= exec_total);

        double const percent =
            1. - (double(exec_total) / double(tfunc_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_idle_rate(
        std::size_t num_thread, bool reset)
    {
        if (num_thread == std::size_t(-1))
            return avg_idle_rate_all(reset);

        std::uint64_t exec_time = exec_times_[num_thread];
        std::uint64_t tfunc_time = tfunc_times_[num_thread];
        std::uint64_t reset_exec_time = reset_idle_rate_time_[num_thread];
        std::uint64_t reset_tfunc_time =
            reset_idle_rate_time_total_[num_thread];

        if (reset)
        {
            reset_idle_rate_time_[num_thread] = exec_time;
            reset_idle_rate_time_total_[num_thread] = tfunc_time;
        }

        HPX_ASSERT(exec_time >= reset_exec_time);
        HPX_ASSERT(tfunc_time >= reset_tfunc_time);

        exec_time -= reset_exec_time;
        tfunc_time -= reset_tfunc_time;

        if (tfunc_time == 0)    // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_time > exec_time);

        double const percent =
            1. - (double(exec_time) / double(tfunc_time));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }
#endif

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_idle_loop_count(
        std::size_t num, bool reset)
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(
                idle_loop_counts_.begin(), idle_loop_counts_.end(), 0ll);
        }
        return idle_loop_counts_[num];
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_busy_loop_count(
        std::size_t num, bool reset)
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(
                busy_loop_counts_.begin(), busy_loop_counts_.end(), 0ll);
        }
        return busy_loop_counts_[num];
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_scheduler_utilization()
        const
    {
        return (std::accumulate(tasks_active_.begin(), tasks_active_.end(),
            std::int64_t(0)) * 100) / thread_count_.load();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::init_perf_counter_data(
        std::size_t pool_threads)
    {
        executed_threads_.resize(pool_threads);
        executed_thread_phases_.resize(pool_threads);

        tfunc_times_.resize(pool_threads);
        exec_times_.resize(pool_threads);

        idle_loop_counts_.resize(pool_threads);
        busy_loop_counts_.resize(pool_threads);

        reset_tfunc_times_.resize(pool_threads);

        tasks_active_.resize(pool_threads);

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        // timestamps/values of last reset operation for various
        // performance counters
        reset_executed_threads_.resize(pool_threads);
        reset_executed_thread_phases_.resize(pool_threads);

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        // timestamps/values of last reset operation for various
        // performance counters
        reset_thread_duration_.resize(pool_threads);
        reset_thread_duration_times_.resize(pool_threads);

        reset_thread_overhead_.resize(pool_threads);
        reset_thread_overhead_times_.resize(pool_threads);
        reset_thread_overhead_times_total_.resize(pool_threads);

        reset_thread_phase_duration_.resize(pool_threads);
        reset_thread_phase_duration_times_.resize(pool_threads);

        reset_thread_phase_overhead_.resize(pool_threads);
        reset_thread_phase_overhead_times_.resize(pool_threads);
        reset_thread_phase_overhead_times_total_.resize(pool_threads);

        reset_cumulative_thread_duration_.resize(pool_threads);

        reset_cumulative_thread_overhead_.resize(pool_threads);
        reset_cumulative_thread_overhead_total_.resize(pool_threads);
#endif
#endif

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        reset_idle_rate_time_.resize(pool_threads);
        reset_idle_rate_time_total_.resize(pool_threads);

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        reset_creation_idle_rate_time_.resize(pool_threads);
        reset_creation_idle_rate_time_total_.resize(pool_threads);

        reset_cleanup_idle_rate_time_.resize(pool_threads);
        reset_cleanup_idle_rate_time_total_.resize(pool_threads);
#endif
#endif
    }
}}}

#endif
