//  Copyright (c) 2017 Shoshana Jakobovits
//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SCHEDULED_THREAD_POOL_IMPL_HPP)
#define HPX_SCHEDULED_THREAD_POOL_IMPL_HPP

#include <hpx/apply.hpp>
#include <hpx/assertion.hpp>
#include <hpx/async.hpp>
#include <hpx/concurrency/barrier.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/topology/topology.hpp>
#include <hpx/state.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/util/yield_while.hpp>

#include <boost/system/system_error.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iosfwd>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct manage_active_thread_count
    {
        manage_active_thread_count(std::atomic<long>& counter)
          : counter_(counter)
        {
        }
        ~manage_active_thread_count()
        {
            --counter_;
        }

        std::atomic<long>& counter_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    struct init_tss_helper
    {
        init_tss_helper(scheduled_thread_pool<Scheduler>& pool,
            std::size_t local_thread_num, std::size_t global_thread_num)
          : pool_(pool)
          , local_thread_num_(local_thread_num)
          , global_thread_num_(global_thread_num)
        {
            pool.notifier_.on_start_thread(local_thread_num_,
                global_thread_num_, pool_.get_pool_id().name().c_str(), "");
            pool.sched_->Scheduler::on_start_thread(local_thread_num_);
        }
        ~init_tss_helper()
        {
            pool_.sched_->Scheduler::on_stop_thread(local_thread_num_);
            pool_.notifier_.on_stop_thread(local_thread_num_, global_thread_num_,
                pool_.get_pool_id().name().c_str(), "");
        }

        scheduled_thread_pool<Scheduler>& pool_;
        std::size_t local_thread_num_;
        std::size_t global_thread_num_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    scheduled_thread_pool<Scheduler>::scheduled_thread_pool(
        std::unique_ptr<Scheduler>
            sched,
        thread_pool_init_parameters const& init)
      : thread_pool_base(init)
      , sched_(std::move(sched))
      , thread_count_(0)
      , tasks_scheduled_(0)
      , network_background_callback_(init.network_background_callback_)
      , max_background_threads_(init.max_background_threads_)
      , max_idle_loop_count_(init.max_idle_loop_count_)
      , max_busy_loop_count_(init.max_busy_loop_count_)
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
        os << "[pool \"" << id_.name() << "\", #" << id_.index()
           << "] with scheduler " << sched_->Scheduler::get_scheduler_name()
           << "\n"
           << "is running on PUs : \n";
        os << std::hex << HPX_CPU_MASK_PREFIX << get_used_processing_units()
           << '\n';
        os << "on numa domains : \n" << get_numa_domain_bitmap() << '\n';
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::report_error(
        std::size_t global_thread_num, std::exception_ptr const& e)
    {
        sched_->Scheduler::set_all_states_at_least(state_terminating);
        this->thread_pool_base::report_error(global_thread_num, e);
        sched_->Scheduler::on_error(global_thread_num, e);
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
        LTM_(info) << "stop: " << id_.name() << " blocking(" << std::boolalpha
                   << blocking << ")";

        if (!threads_.empty())
        {
            // wake up if suspended
            resume_internal(blocking, throws);

            // set state to stopping
            sched_->Scheduler::set_all_states_at_least(state_stopping);

            // make sure we're not waiting
            sched_->Scheduler::do_some_work(std::size_t(-1));

            if (blocking)
            {
                for (std::size_t i = 0; i != threads_.size(); ++i)
                {
                    // skip this if already stopped
                    if (!threads_[i].joinable())
                        continue;

                    // make sure no OS thread is waiting
                    LTM_(info) << "stop: " << id_.name() << " notify_all";

                    sched_->Scheduler::do_some_work(std::size_t(-1));

                    LTM_(info) << "stop: " << id_.name() << " join:" << i;

                    {
                        // unlock the lock while joining
                        util::unlock_guard<Lock> ul(l);
                        remove_processing_unit_internal(i);
                    }
                }
                threads_.clear();
            }
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::stop(
        std::unique_lock<std::mutex>& l, bool blocking)
    {
        HPX_ASSERT(l.owns_lock());
        return stop_locked(l, blocking);
    }

    template <typename Scheduler>
    bool hpx::threads::detail::scheduled_thread_pool<Scheduler>::run(
        std::unique_lock<std::mutex>& l, std::size_t pool_threads)
    {
        HPX_ASSERT(l.owns_lock());

        LTM_(info)    //-V128
            << "run: " << id_.name()
            << " number of processing units available: "    //-V128
            << threads::hardware_concurrency();
        LTM_(info)    //-V128
            << "run: " << id_.name() << " creating " << pool_threads
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

        LTM_(info) << "run: " << id_.name()
                   << " timestamp_scale: " << timestamp_scale_;    //-V128

        // run threads and wait for initialization to complete
        std::size_t thread_num = 0;
        std::shared_ptr<util::barrier> startup =
            std::make_shared<util::barrier>(pool_threads + 1);
        try
        {
            topology const& topo = create_topology();

            for (/**/; thread_num != pool_threads; ++thread_num)
            {
                std::size_t global_thread_num = this->thread_offset_ + thread_num;
                threads::mask_cref_type mask =
                    affinity_data_.get_pu_mask(topo, global_thread_num);

                // thread_num ordering: 1. threads of default pool
                //                      2. threads of first special pool
                //                      3. etc.
                // get_pu_mask expects index according to ordering of masks
                // in affinity_data::affinity_masks_
                // which is in order of occupied PU
                LTM_(info)    //-V128
                    << "run: " << id_.name() << " create OS thread "
                    << global_thread_num    //-V128
                    << ": will run on processing units within this mask: "
                    << std::hex << HPX_CPU_MASK_PREFIX << mask;

                // create a new thread
                add_processing_unit_internal(
                    thread_num, global_thread_num, startup);
            }

            // wait for all threads to have started up
            startup->wait();

            HPX_ASSERT(pool_threads == std::size_t(thread_count_.load()));
        }
        catch (std::exception const& e)
        {
            LTM_(always) << "run: " << id_.name()
                         << " failed with: " << e.what();

            // trigger the barrier
            pool_threads -= (thread_num + 1);
            while (pool_threads-- != 0)
                startup->wait();

            stop_locked(l);
            threads_.clear();

            return false;
        }

        LTM_(info) << "run: " << id_.name() << " running";
        return true;
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_internal(bool blocking,
        error_code& ec)
    {
        for (std::size_t virt_core = 0; virt_core != threads_.size();
             ++virt_core)
        {
            this->sched_->Scheduler::resume(virt_core);
        }

        if (blocking)
        {
            for (std::size_t virt_core = 0; virt_core != threads_.size();
                ++virt_core)
            {
                if (threads_[virt_core].joinable())
                {
                    resume_processing_unit_direct(virt_core, ec);
                }
            }
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_direct(error_code& ec)
    {
        this->resume_internal(true, ec);
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_internal(error_code& ec)
    {
        util::yield_while([this]()
            {
                return this->sched_->Scheduler::get_thread_count() >
                    this->get_background_thread_count();
            }, "scheduled_thread_pool::suspend_internal",
            hpx::threads::pending);

        for (std::size_t i = 0; i != threads_.size(); ++i)
        {
            hpx::state expected = state_running;
            sched_->Scheduler::get_state(i).compare_exchange_strong(expected,
                 state_pre_sleep);
        }

        for (std::size_t i = 0; i != threads_.size(); ++i)
        {
            suspend_processing_unit_direct(i, ec);
        }
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_direct(error_code& ec)
    {
        if (threads::get_self_ptr() && hpx::this_thread::get_pool() == this)
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "scheduled_thread_pool<Scheduler>::suspend_direct",
                "cannot suspend a pool from itself");
            return;
        }

        this->suspend_internal(ec);
    }

    template <typename Scheduler>
    void hpx::threads::detail::scheduled_thread_pool<Scheduler>::thread_func(
        std::size_t thread_num, std::size_t global_thread_num,
        std::shared_ptr<util::barrier> startup)
    {
        topology const& topo = create_topology();

        // Set the affinity for the current thread.
        threads::mask_cref_type mask =
            affinity_data_.get_pu_mask(topo, global_thread_num);

        if (LHPX_ENABLED(debug))
            topo.write_to_log();

        error_code ec(lightweight);
        if (any(mask))
        {
            topo.set_thread_affinity_mask(mask, ec);
            if (ec)
            {
                LTM_(warning)    //-V128
                    << "thread_func: " << id_.name()
                    << " setting thread affinity on OS thread "    //-V128
                    << global_thread_num
                    << " failed with: " << ec.get_message();
            }
        }
        else
        {
            LTM_(debug)    //-V128
                << "thread_func: " << id_.name()
                << " setting thread affinity on OS thread "    //-V128
                << global_thread_num << " was explicitly disabled.";
        }

        // Setting priority of worker threads to a lower priority, this
        // needs to
        // be done in order to give the parcel pool threads higher
        // priority
        if (mode_ & policies::reduce_thread_priority)
        {
            topo.reduce_thread_priority(ec);
            if (ec)
            {
                LTM_(warning)    //-V128
                    << "thread_func: " << id_.name()
                    << " reducing thread priority on OS thread "    //-V128
                    << global_thread_num
                    << " failed with: " << ec.get_message();
            }
        }

        // manage the number of this thread in its TSS
        init_tss_helper<Scheduler> tss_helper(
            *this, thread_num, global_thread_num);

        ++thread_count_;

        // set state to running
        std::atomic<hpx::state>& state =
            sched_->Scheduler::get_state(thread_num);
        hpx::state oldstate = state.exchange(state_running);
        HPX_ASSERT(oldstate <= state_running);
        HPX_UNUSED(oldstate);

        // wait for all threads to start up before before starting HPX work
        startup->wait();

        LTM_(info)    //-V128
            << "thread_func: " << id_.name()
            << " starting OS thread: " << thread_num;    //-V128

        try
        {
            try
            {
                manage_active_thread_count count(thread_count_);

                // run the work queue
                hpx::threads::coroutines::prepare_main_thread main_thread;
                HPX_UNUSED(main_thread);

                // run main Scheduler loop until terminated
                scheduling_counter_data& counter_data =
                    counter_data_[thread_num];

                detail::scheduling_counters counters(
                    counter_data.executed_threads_,
                    counter_data.executed_thread_phases_,
                    counter_data.tfunc_times_,
                    counter_data.exec_times_,
                    counter_data.idle_loop_counts_,
                    counter_data.busy_loop_counts_,
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
                    counter_data.tasks_active_,
                    counter_data.background_duration_,
                    counter_data.background_send_duration_,
                    counter_data.background_receive_duration_);
#else
                    counter_data.tasks_active_);
#endif // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

                detail::scheduling_callbacks callbacks(
                    util::deferred_call(    //-V107
                        &policies::scheduler_base::idle_callback, sched_.get(),
                        thread_num),
                    nullptr, nullptr, max_background_threads_,
                    max_idle_loop_count_, max_busy_loop_count_);

                if ((mode_ & policies::do_background_work) &&
                    network_background_callback_)
                {
#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
                    callbacks.background_ = util::deferred_call(    //-V107
                        network_background_callback_,
                        global_thread_num,
                        std::ref(counter_data.background_send_duration_),
                        std::ref(counter_data.background_receive_duration_));
#else
                    callbacks.background_ = util::deferred_call(    //-V107
                        network_background_callback_,
                        global_thread_num);
#endif
                }

                sched_->Scheduler::set_scheduler_mode(mode_);
                detail::scheduling_loop(
                    thread_num, *sched_, counters, callbacks);

                // the OS thread is allowed to exit only if no more HPX
                // threads exist or if some other thread has terminated
                HPX_ASSERT(
                    (sched_->Scheduler::get_thread_count(
                        suspended, thread_priority_default, thread_num) == 0 &&
                     sched_->Scheduler::get_queue_length(thread_num) == 0) ||
                    sched_->Scheduler::get_state(thread_num) > state_stopping);
            }
            catch (hpx::exception const& e)
            {
                LFATAL_    //-V128
                    << "thread_func: " << id_.name()
                    << " thread_num:" << global_thread_num    //-V128
                    << " : caught hpx::exception: " << e.what()
                    << ", aborted thread execution";

                report_error(global_thread_num, std::current_exception());
                return;
            }
            catch (boost::system::system_error const& e)
            {
                LFATAL_    //-V128
                    << "thread_func: " << id_.name()
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
                << "thread_func: " << id_.name()
                << " thread_num:" << global_thread_num    //-V128
                << " : caught unexpected "                //-V128
                    "exception, aborted thread execution";

            report_error(global_thread_num, std::current_exception());
            return;
        }

        LTM_(info)    //-V128
            << "thread_func: " << id_.name()
            << " thread_num: " << global_thread_num
            << " , ending OS thread, executed "    //-V128
            << counter_data_[global_thread_num].executed_threads_
            << " HPX threads";
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

        // update statistics
        ++tasks_scheduled_;
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

        // update statistics
        ++tasks_scheduled_;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_state scheduled_thread_pool<Scheduler>::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state(id, new_state,    //-V107
            new_state_ex, priority,
            thread_schedule_hint(
                static_cast<std::int16_t>(get_worker_thread_num())),
            true, ec);
    }

    template <typename Scheduler>
    thread_id_type scheduled_thread_pool<Scheduler>::set_state(
        util::steady_time_point const& abs_time, thread_id_type const& id,
        thread_state_enum newstate, thread_state_ex_enum newstate_ex,
        thread_priority priority, error_code& ec)
    {
        return detail::set_thread_state_timed(*sched_, abs_time, id, newstate,
            newstate_ex, priority,
            thread_schedule_hint(
                static_cast<std::int16_t>(get_worker_thread_num())),
            nullptr, true, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // performance counters
    template <typename InIter, typename OutIter, typename ProjSrc,
        typename ProjDest>
    OutIter copy_projected(InIter first, InIter last, OutIter dest,
        ProjSrc&& srcproj, ProjDest&& destproj)
    {
        while (first != last)
        {
            util::invoke(destproj, *dest++) = util::invoke(srcproj, *first++);
        }
        return dest;
    }

    template <typename InIter, typename T, typename Proj>
    T accumulate_projected(InIter first, InIter last, T init, Proj && proj)
    {
        while (first != last)
        {
            init = std::move(init) + util::invoke(proj, *first++);
        }
        return init;
    }

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_threads(
        std::size_t num, bool reset)
    {
        std::int64_t executed_threads = 0;
        std::int64_t reset_executed_threads = 0;

        if (num != std::size_t(-1))
        {
            executed_threads = counter_data_[num].executed_threads_;
            reset_executed_threads = counter_data_[num].reset_executed_threads_;

            if (reset)
                counter_data_[num].reset_executed_threads_ = executed_threads;
        }
        else
        {
            executed_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_threads_);
            reset_executed_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_executed_threads_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_threads_,
                    &scheduling_counter_data::reset_executed_threads_);
            }
        }

        HPX_ASSERT(executed_threads >= reset_executed_threads);

        return executed_threads - reset_executed_threads;
    }
#endif

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_threads() const
    {
        std::int64_t executed_threads =
            accumulate_projected(counter_data_.begin(), counter_data_.end(),
                std::int64_t(0), &scheduling_counter_data::executed_threads_);

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        std::int64_t reset_executed_threads = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_executed_threads_);

        HPX_ASSERT(executed_threads >= reset_executed_threads);
        return executed_threads - reset_executed_threads;
#else
        return executed_threads;
#endif
    }

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_executed_thread_phases(
        std::size_t num, bool reset)
    {
        std::int64_t executed_phases = 0;
        std::int64_t reset_executed_phases = 0;

        if (num != std::size_t(-1))
        {
            executed_phases = counter_data_[num].executed_thread_phases_;
            reset_executed_phases =
                counter_data_[num].reset_executed_thread_phases_;

            if (reset)
                counter_data_[num].reset_executed_thread_phases_ =
                    executed_phases;
        }
        else
        {
            executed_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_thread_phases_);
            reset_executed_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_executed_thread_phases_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_executed_thread_phases_);
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
        std::int64_t exec_total = 0;
        std::int64_t num_phases = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t reset_num_phases = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            num_phases = counter_data_[num].executed_thread_phases_;

            reset_exec_total =
                counter_data_[num].reset_thread_phase_duration_times_;
            reset_num_phases = counter_data_[num].reset_thread_phase_duration_;

            if (reset)
            {
                counter_data_[num].reset_thread_phase_duration_ = num_phases;
                counter_data_[num].reset_thread_phase_duration_times_ =
                    exec_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            num_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_thread_phases_);

            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_phase_duration_times_);
            reset_num_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_phase_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_phase_duration_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_phase_duration_);
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(num_phases >= reset_num_phases);

        exec_total -= reset_exec_total;
        num_phases -= reset_num_phases;

        return std::int64_t(
                (double(exec_total) * timestamp_scale_) / double(num_phases)
            );
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_duration(
        std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t num_threads = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t reset_num_threads = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            num_threads = counter_data_[num].executed_threads_;

            reset_exec_total = counter_data_[num].reset_thread_duration_times_;
            reset_num_threads = counter_data_[num].reset_thread_duration_;

            if (reset)
            {
                counter_data_[num].reset_thread_duration_ = num_threads;
                counter_data_[num].reset_thread_duration_times_ = exec_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            num_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_threads_);

            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_duration_times_);
            reset_num_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_duration_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_threads_,
                    &scheduling_counter_data::reset_thread_duration_);
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(num_threads >= reset_num_threads);

        exec_total -= reset_exec_total;
        num_threads -= reset_num_threads;

        return std::int64_t(
            (double(exec_total) * timestamp_scale_) / double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_phase_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t num_phases = 0;

        std::int64_t reset_exec_total = 0;
        std::int64_t reset_tfunc_total = 0;
        std::int64_t reset_num_phases = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            tfunc_total = counter_data_[num].tfunc_times_;
            num_phases = counter_data_[num].executed_thread_phases_;

            reset_exec_total =
                counter_data_[num].reset_thread_phase_overhead_times_;
            reset_tfunc_total =
                counter_data_[num].reset_thread_phase_overhead_times_total_;
            reset_num_phases = counter_data_[num].reset_thread_phase_overhead_;

            if (reset)
            {
                counter_data_[num].reset_thread_phase_overhead_times_ =
                    exec_total;
                counter_data_[num].reset_thread_phase_overhead_times_total_ =
                    tfunc_total;
                counter_data_[num].reset_thread_phase_overhead_ = num_phases;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            num_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_thread_phases_);

            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_phase_overhead_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::
                    reset_thread_phase_overhead_times_total_);
            reset_num_phases = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_phase_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::
                        reset_thread_phase_overhead_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::
                        reset_thread_phase_overhead_times_total_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_phase_overhead_);
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

        return std::int64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) /
            double(num_phases));
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_thread_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t num_threads = 0;

        std::int64_t reset_exec_total = 0;
        std::int64_t reset_tfunc_total = 0;
        std::int64_t reset_num_threads = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            tfunc_total = counter_data_[num].tfunc_times_;
            num_threads = counter_data_[num].executed_threads_;

            reset_exec_total = counter_data_[num].reset_thread_overhead_times_;
            reset_tfunc_total =
                counter_data_[num].reset_thread_overhead_times_total_;
            reset_num_threads = counter_data_[num].reset_thread_overhead_;

            if (reset)
            {
                counter_data_[num].reset_thread_overhead_times_ = exec_total;
                counter_data_[num].reset_thread_overhead_times_total_ =
                    tfunc_total;
                counter_data_[num].reset_thread_overhead_ = num_threads;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            num_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::executed_threads_);

            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_overhead_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_overhead_times_total_);
            reset_num_threads = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_thread_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_thread_overhead_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::
                        reset_thread_overhead_times_total_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::executed_thread_phases_,
                    &scheduling_counter_data::reset_thread_overhead_);
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

        return std::int64_t(
            double((tfunc_total - exec_total) * timestamp_scale_) /
            double(num_threads));
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_thread_duration(
        std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t reset_exec_total = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            reset_exec_total =
                counter_data_[num].reset_cumulative_thread_duration_;

            if (reset)
            {
                counter_data_[num].reset_cumulative_thread_duration_ =
                    exec_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_cumulative_thread_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_cumulative_thread_duration_);
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);

        exec_total -= reset_exec_total;

        return std::int64_t(double(exec_total) * timestamp_scale_);
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_cumulative_thread_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t exec_total = 0;
        std::int64_t reset_exec_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            exec_total = counter_data_[num].exec_times_;
            tfunc_total = counter_data_[num].tfunc_times_;

            reset_exec_total =
                counter_data_[num].reset_cumulative_thread_overhead_;
            reset_tfunc_total =
                counter_data_[num].reset_cumulative_thread_overhead_total_;

            if (reset)
            {
                counter_data_[num].reset_cumulative_thread_overhead_ =
                    exec_total;
                counter_data_[num].reset_cumulative_thread_overhead_total_ =
                    tfunc_total;
            }
        }
        else
        {
            exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::exec_times_);
            reset_exec_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_cumulative_thread_overhead_);

            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::
                    reset_cumulative_thread_overhead_total_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::exec_times_,
                    &scheduling_counter_data::reset_cumulative_thread_overhead_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::
                        reset_cumulative_thread_overhead_total_);
            }
        }

        HPX_ASSERT(exec_total >= reset_exec_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        exec_total -= reset_exec_total;
        tfunc_total -= reset_tfunc_total;

        return std::int64_t(
            (double(tfunc_total) - double(exec_total)) * timestamp_scale_);
    }
#endif // HPX_HAVE_THREAD_IDLE_RATES
#endif // HPX_HAVE_THREAD_CUMULATIVE_COUNTS

#if defined(HPX_HAVE_BACKGROUND_THREAD_COUNTERS) && defined(HPX_HAVE_THREAD_IDLE_RATES)
    ////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_background_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            tfunc_total = counter_data_[num].tfunc_times_;
            reset_tfunc_total = counter_data_[num].reset_background_tfunc_times_;

            bg_total = counter_data_[num].background_duration_;
            reset_bg_total = counter_data_[num].reset_background_overhead_;

            if (reset)
            {
                counter_data_[num].reset_background_overhead_ = bg_total;
                counter_data_[num].reset_background_tfunc_times_ = tfunc_total;
            }
        }
        else
        {
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_tfunc_times_);

            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_background_tfunc_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_duration_,
                    &scheduling_counter_data::reset_background_overhead_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        if (tfunc_total == 0)    // avoid division by zero
            return 1000LL;

        tfunc_total -= reset_tfunc_total;
        bg_total -= reset_bg_total;

        // this is now a 0.1 %
        return std::int64_t((double(bg_total) / tfunc_total) * 1000);
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_background_work_duration(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;

        if (num != std::size_t(-1))
        {
            bg_total = counter_data_[num].background_duration_;
            reset_bg_total = counter_data_[num].reset_background_duration_;

            if (reset)
            {
                counter_data_[num].reset_background_duration_ = bg_total;
            }
        }
        else
        {
            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_duration_,
                    &scheduling_counter_data::reset_background_duration_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        bg_total -= reset_bg_total;
        return std::int64_t(double(bg_total) * timestamp_scale_);
    }

    ////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_background_send_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            tfunc_total = counter_data_[num].tfunc_times_;
            reset_tfunc_total = counter_data_[num].reset_background_send_tfunc_times_;

            bg_total = counter_data_[num].background_send_duration_;
            reset_bg_total = counter_data_[num].reset_background_send_overhead_;

            if (reset)
            {
                counter_data_[num].reset_background_send_overhead_ = bg_total;
                counter_data_[num].reset_background_send_tfunc_times_ = tfunc_total;
            }
        }
        else
        {
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_send_tfunc_times_);

            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_send_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_send_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_background_send_tfunc_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_send_duration_,
                    &scheduling_counter_data::reset_background_send_overhead_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        if (tfunc_total == 0)    // avoid division by zero
            return 1000LL;

        tfunc_total -= reset_tfunc_total;
        bg_total -= reset_bg_total;

        // this is now a 0.1 %
        return std::int64_t((double(bg_total) / tfunc_total) * 1000);
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_background_send_duration(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;

        if (num != std::size_t(-1))
        {
            bg_total = counter_data_[num].background_send_duration_;
            reset_bg_total = counter_data_[num].reset_background_send_duration_;

            if (reset)
            {
                counter_data_[num].reset_background_send_duration_ = bg_total;
            }
        }
        else
        {
            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_send_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_send_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_send_duration_,
                    &scheduling_counter_data::reset_background_send_duration_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        bg_total -= reset_bg_total;
        return std::int64_t(double(bg_total) * timestamp_scale_);
    }

    ////////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_background_receive_overhead(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            tfunc_total = counter_data_[num].tfunc_times_;
            reset_tfunc_total =
                counter_data_[num].reset_background_receive_tfunc_times_;

            bg_total = counter_data_[num].background_receive_duration_;
            reset_bg_total =
                counter_data_[num].reset_background_receive_overhead_;

            if (reset)
            {
                counter_data_[num].reset_background_receive_overhead_ =
                    bg_total;
                counter_data_[num].reset_background_receive_tfunc_times_ =
                    tfunc_total;
            }
        }
        else
        {
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_receive_tfunc_times_);

            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_receive_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_receive_overhead_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_background_receive_tfunc_times_);
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_receive_duration_,
                    &scheduling_counter_data::reset_background_receive_overhead_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        if (tfunc_total == 0)    // avoid division by zero
            return 1000LL;

        tfunc_total -= reset_tfunc_total;
        bg_total -= reset_bg_total;

        // this is now a 0.1 %
        return std::int64_t((double(bg_total) / tfunc_total) * 1000);
    }

    template <typename Scheduler>
    std::int64_t
    scheduled_thread_pool<Scheduler>::get_background_receive_duration(
        std::size_t num, bool reset)
    {
        std::int64_t bg_total = 0;
        std::int64_t reset_bg_total = 0;

        if (num != std::size_t(-1))
        {
            bg_total = counter_data_[num].background_receive_duration_;
            reset_bg_total = counter_data_[num].reset_background_receive_duration_;

            if (reset)
            {
                counter_data_[num].reset_background_receive_duration_ = bg_total;
            }
        }
        else
        {
            bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::background_receive_duration_);
            reset_bg_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_background_receive_duration_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::background_receive_duration_,
                    &scheduling_counter_data::reset_background_receive_duration_);
            }
        }

        HPX_ASSERT(bg_total >= reset_bg_total);
        bg_total -= reset_bg_total;
        return std::int64_t(double(bg_total) * timestamp_scale_);
    }
#endif    // HPX_HAVE_BACKGROUND_THREAD_COUNTERS

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_cumulative_duration(
        std::size_t num, bool reset)
    {
        std::int64_t tfunc_total = 0;
        std::int64_t reset_tfunc_total = 0;

        if (num != std::size_t(-1))
        {
            tfunc_total = counter_data_[num].tfunc_times_;
            reset_tfunc_total = counter_data_[num].reset_tfunc_times_;

            if (reset)
                counter_data_[num].reset_tfunc_times_ = tfunc_total;
        }
        else
        {
            tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::tfunc_times_);
            reset_tfunc_total = accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::reset_tfunc_times_);

            if (reset)
            {
                copy_projected(counter_data_.begin(), counter_data_.end(),
                    counter_data_.begin(),
                    &scheduling_counter_data::tfunc_times_,
                    &scheduling_counter_data::reset_tfunc_times_);
            }
        }

        HPX_ASSERT(tfunc_total >= reset_tfunc_total);

        tfunc_total -= reset_tfunc_total;

        return std::int64_t(double(tfunc_total) * timestamp_scale_);
    }

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_creation_idle_rate(
        std::size_t, bool reset)
    {
        double const creation_total =
            static_cast<double>(sched_->Scheduler::get_creation_time(reset));

        std::int64_t exec_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_creation_idle_rate_time_);
        std::int64_t reset_tfunc_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_creation_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_creation_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_creation_idle_rate_time_);
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

        std::int64_t exec_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_cleanup_idle_rate_time_);
        std::int64_t reset_tfunc_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_cleanup_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_cleanup_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_cleanup_idle_rate_time_);
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
#endif // HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::avg_idle_rate_all(bool reset)
    {
        std::int64_t exec_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::exec_times_);
        std::int64_t tfunc_total = accumulate_projected(counter_data_.begin(),
            counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::tfunc_times_);

        std::int64_t reset_exec_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_idle_rate_time_);
        std::int64_t reset_tfunc_total = accumulate_projected(
            counter_data_.begin(), counter_data_.end(), std::int64_t(0),
            &scheduling_counter_data::reset_idle_rate_time_total_);

        if (reset)
        {
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::exec_times_,
                &scheduling_counter_data::reset_idle_rate_time_);
            copy_projected(counter_data_.begin(), counter_data_.end(),
                counter_data_.begin(),
                &scheduling_counter_data::tfunc_times_,
                &scheduling_counter_data::reset_idle_rate_time_total_);
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
        std::size_t num, bool reset)
    {
        if (num == std::size_t(-1))
            return avg_idle_rate_all(reset);

        std::int64_t exec_time = counter_data_[num].exec_times_;
        std::int64_t tfunc_time = counter_data_[num].tfunc_times_;
        std::int64_t reset_exec_time =
            counter_data_[num].reset_idle_rate_time_;
        std::int64_t reset_tfunc_time =
            counter_data_[num].reset_idle_rate_time_total_;

        if (reset)
        {
            counter_data_[num].reset_idle_rate_time_ = exec_time;
            counter_data_[num].reset_idle_rate_time_total_ = tfunc_time;
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
#endif // HPX_HAVE_THREAD_IDLE_RATES

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_idle_loop_count(
        std::size_t num, bool reset)
    {
        if (num == std::size_t(-1))
        {
            return accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::idle_loop_counts_);
        }
        return counter_data_[num].idle_loop_counts_;
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_busy_loop_count(
        std::size_t num, bool reset)
    {
        if (num == std::size_t(-1))
        {
            return accumulate_projected(counter_data_.begin(),
                counter_data_.end(), std::int64_t(0),
                &scheduling_counter_data::busy_loop_counts_);
        }
        return counter_data_[num].busy_loop_counts_;
    }

    template <typename Scheduler>
    std::int64_t scheduled_thread_pool<Scheduler>::get_scheduler_utilization()
        const
    {
        return (accumulate_projected(counter_data_.begin(), counter_data_.end(),
                    std::int64_t(0), &scheduling_counter_data::tasks_active_) *
                   100) /
            thread_count_.load();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::init_perf_counter_data(
        std::size_t pool_threads)
    {
        counter_data_.resize(pool_threads);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::size_t
    scheduled_thread_pool<Scheduler>::get_policy_element(executor_parameter p,
        error_code& ec) const
    {
        if (&ec != &throws)
            ec = make_success_code();

        switch(p) {
        case threads::detail::min_concurrency:
//             return min_punits_;
            break;

        case threads::detail::max_concurrency:
//             return max_punits_;
            break;

        case threads::detail::current_concurrency:
            return thread_count_;

        default:
            break;
        }

        HPX_THROWS_IF(ec, bad_parameter,
            "thread_pool_executor::get_policy_element",
            "requested value of invalid policy element");
        return std::size_t(-1);
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::get_statistics(
        executor_statistics& s, error_code& ec) const
    {
        s.queue_length_ = sched_->Scheduler::get_queue_length();
        s.tasks_scheduled_ = tasks_scheduled_;
        s.tasks_completed_ = get_executed_threads();

        if (&ec != &throws)
            ec = make_success_code();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::add_processing_unit_internal(
        std::size_t virt_core, std::size_t thread_num,
        std::shared_ptr<util::barrier> startup, error_code& ec)
    {
        std::unique_lock<typename Scheduler::pu_mutex_type>
            l(sched_->Scheduler::get_pu_mutex(virt_core));

        if (threads_.size() <= virt_core)
            threads_.resize(virt_core + 1);

        if (threads_[virt_core].joinable())
        {
            l.unlock();
            HPX_THROWS_IF(ec, bad_parameter,
                "scheduled_thread_pool<Scheduler>::add_processing_unit",
                "the given virtual core has already been added to this "
                "thread pool");
            return;
        }

        std::atomic<hpx::state>& state =
            sched_->Scheduler::get_state(virt_core);
        hpx::state oldstate = state.exchange(state_initialized);
        HPX_ASSERT(oldstate == state_stopped || oldstate == state_initialized);
        HPX_UNUSED(oldstate);

        threads_[virt_core] =
            std::thread(&scheduled_thread_pool::thread_func, this,
                virt_core, thread_num, std::move(startup));

        if (&ec != &throws)
            ec = make_success_code();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::add_processing_unit(
        std::size_t virt_core, std::size_t thread_num, error_code& ec)
    {
        if (!(mode_ & threads::policies::enable_elasticity))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "scheduled_thread_pool<Scheduler>::add_processing_unit",
                "this thread pool does not support dynamically adding "
                "processing units");
        }

        std::shared_ptr<util::barrier> startup =
            std::make_shared<util::barrier>(2);

        add_processing_unit_internal(virt_core, thread_num, startup, ec);

        startup->wait();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::remove_processing_unit(
        std::size_t virt_core, error_code& ec)
    {
        if (!(mode_ & threads::policies::enable_elasticity))
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "scheduled_thread_pool<Scheduler>::remove_processing_unit",
                "this thread pool does not support dynamically removing "
                "processing units");
        }

        remove_processing_unit_internal(virt_core, ec);
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::remove_processing_unit_internal(
        std::size_t virt_core, error_code& ec)
    {
        std::unique_lock<typename Scheduler::pu_mutex_type>
            l(sched_->Scheduler::get_pu_mutex(virt_core));

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            HPX_THROWS_IF(ec, bad_parameter,
                "scheduled_thread_pool<Scheduler>::remove_processing_unit",
                "the given virtual core has already been stopped to run on "
                "this thread pool");
            return;
        }

        std::atomic<hpx::state>& state =
            sched_->Scheduler::get_state(virt_core);

        // inform the scheduler to stop the virtual core
        hpx::state oldstate = state.exchange(state_stopping);

        if (oldstate > state_stopping)
        {
            // If thread was terminating or already stopped we don't want to
            // change the value back to stopping, so we restore the old state.
            state.store(oldstate);
        }

        HPX_ASSERT(oldstate == state_starting ||
            oldstate == state_running || oldstate == state_stopping ||
            oldstate == state_stopped || oldstate == state_terminating);

        std::thread t;
        std::swap(threads_[virt_core], t);

        l.unlock();

        if (threads::get_self_ptr() && this == hpx::this_thread::get_pool())
        {
            std::size_t thread_num = thread_offset_ + virt_core;

            util::yield_while([thread_num]()
                {
                    return thread_num == hpx::get_worker_thread_num();
                }, "scheduled_thread_pool::remove_processing_unit_internal",
                hpx::threads::pending);
        }

        t.join();
    }

    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::suspend_processing_unit_direct(
        std::size_t virt_core, error_code& ec)
    {
        // Yield to other HPX threads if lock is not available to avoid
        // deadlocks when multiple HPX threads try to resume or suspend pus.
        std::unique_lock<typename Scheduler::pu_mutex_type>
            l(sched_->Scheduler::get_pu_mutex(virt_core), std::defer_lock);

        util::yield_while(
            [&l]()
            {
                return !l.try_lock();
            }, "scheduled_thread_pool::suspend_processing_unit_direct",
            hpx::threads::pending);

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            HPX_THROWS_IF(ec, bad_parameter,
                "scheduled_thread_pool<Scheduler>::suspend_processing_unit_"
                "direct",
                "the given virtual core has already been stopped to run on "
                "this thread pool");
            return;
        }

        std::atomic<hpx::state>& state =
            sched_->Scheduler::get_state(virt_core);

        // Inform the scheduler to suspend the virtual core only if running
        hpx::state expected = state_running;
        state.compare_exchange_strong(expected, state_pre_sleep);

        l.unlock();

        HPX_ASSERT(expected == state_running || expected == state_pre_sleep ||
            expected == state_sleeping);

        util::yield_while([&state]()
            {
                return state.load() == state_pre_sleep;
            }, "scheduled_thread_pool::suspend_processing_unit_direct",
            hpx::threads::pending);
    }
    template <typename Scheduler>
    void scheduled_thread_pool<Scheduler>::resume_processing_unit_direct(
        std::size_t virt_core, error_code& ec)
    {
        // Yield to other HPX threads if lock is not available to avoid
        // deadlocks when multiple HPX threads try to resume or suspend pus.
        std::unique_lock<typename Scheduler::pu_mutex_type>
            l(sched_->Scheduler::get_pu_mutex(virt_core), std::defer_lock);
        util::yield_while([&l]()
            {
                return !l.try_lock();
            }, "scheduled_thread_pool::resume_processing_unit_direct",
            hpx::threads::pending);

        if (threads_.size() <= virt_core || !threads_[virt_core].joinable())
        {
            l.unlock();
            HPX_THROWS_IF(ec, bad_parameter,
                "scheduled_thread_pool<Scheduler>::resume_processing_unit",
                "the given virtual core has already been stopped to run on "
                "this thread pool");
            return;
        }

        l.unlock();

        std::atomic<hpx::state>& state =
            sched_->Scheduler::get_state(virt_core);

        util::yield_while([this, &state, virt_core]()
            {
                this->sched_->Scheduler::resume(virt_core);
                return state.load() == state_sleeping;
            }, "scheduled_thread_pool::resume_processing_unit_direct",
            hpx::threads::pending);
    }
}}}

#endif
