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
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/thread_num_tss.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/atomic.hpp>
#include <boost/exception_ptr.hpp>
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
    template <typename Scheduler>
    void thread_pool<Scheduler>::init_tss(std::size_t num)
    {
        thread_num_tss_.init_tss(num);
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::deinit_tss()
    {
        thread_num_tss_.deinit_tss();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    thread_pool<Scheduler>::thread_pool(Scheduler& sched,
            threads::policies::callback_notifier& notifier,
            char const* pool_name, policies::scheduler_mode m)
      : sched_(sched),
        notifier_(notifier),
        pool_name_(pool_name),
        thread_count_(0),
        used_processing_units_(),
        mode_(m)
    {
        timestamp_scale_ = 1.0;
    }

    template <typename Scheduler>
    thread_pool<Scheduler>::~thread_pool()
    {
        if (!threads_.empty()) {
            if (!sched_.has_reached_state(state_suspended))
            {
                // still running
                lcos::local::no_mutex mtx;
                std::unique_lock<lcos::local::no_mutex> l(mtx);
                stop_locked(l);
            }
            threads_.clear();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    hpx::state thread_pool<Scheduler>::get_state() const
    {
        // get_worker_thread_num returns the global thread number which might
        // be too large. This function might get called from within
        // background_work inside the os executors
        if (thread_count_ != 0)
        {
            std::size_t num_thread = get_worker_thread_num() % thread_count_;
            if (num_thread != std::size_t(-1))
                return get_state(num_thread);
        }
        return sched_.get_minmax_state().second;
    }

    template <typename Scheduler>
    hpx::state thread_pool<Scheduler>::get_state(std::size_t num_thread) const
    {
        HPX_ASSERT(num_thread != std::size_t(-1));
        return sched_.get_state(num_thread).load();
    }

    template <typename Scheduler>
    bool thread_pool<Scheduler>::has_reached_state(hpx::state s) const
    {
        return sched_.has_reached_state(s);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::size_t thread_pool<Scheduler>::init(std::size_t num_threads,
        policies::init_affinity_data const& data)
    {
        topology const& topology_ = get_topology();
        std::size_t cores_used = sched_.Scheduler::init(data, topology_);

        resize(used_processing_units_, threads::hardware_concurrency());
        for (std::size_t i = 0; i != num_threads; ++i)
            used_processing_units_ |= sched_.Scheduler::get_pu_mask(topology_, i);

        return cores_used;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::size_t thread_pool<Scheduler>::get_pu_num(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_pu_num(num_thread);
    }

    template <typename Scheduler>
    mask_cref_type thread_pool<Scheduler>::get_pu_mask(
        topology const& topology, std::size_t num_thread) const
    {
        return sched_.Scheduler::get_pu_mask(topology, num_thread);
    }

    template <typename Scheduler>
    mask_cref_type thread_pool<Scheduler>::get_used_processing_units() const
    {
        return used_processing_units_;
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::do_some_work(std::size_t num_thread)
    {
        sched_.Scheduler::do_some_work(num_thread);
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::report_error(std::size_t num,
        boost::exception_ptr const& e)
    {
        sched_.set_all_states(state_terminating);
        notifier_.on_error(num, e);
        sched_.Scheduler::on_error(num, e);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void thread_pool<Scheduler>::create_thread(thread_init_data& data,
        thread_id_type& id, thread_state_enum initial_state, bool run_now,
        error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_.is_state(state_running))
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "thread_pool<Scheduler>::create_thread",
                "invalid state: thread pool is not running");
            return;
        }

        detail::create_thread(&sched_, data, id, initial_state, run_now, ec); //-V601
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::create_work(thread_init_data& data,
        thread_state_enum initial_state, error_code& ec)
    {
        // verify state
        if (thread_count_ == 0 && !sched_.is_state(state_running))
        {
            // thread-manager is not currently running
            HPX_THROWS_IF(ec, invalid_status,
                "thread_pool<Scheduler>::create_work",
                "invalid state: thread pool is not running");
            return;
        }

        detail::create_work(&sched_, data, initial_state, ec); //-V601
    }

    template <typename Scheduler>
    thread_state thread_pool<Scheduler>::set_state(
        thread_id_type const& id, thread_state_enum new_state,
        thread_state_ex_enum new_state_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state(id, new_state, //-V107
            new_state_ex, priority, get_worker_thread_num(), ec);
    }

    template <typename Scheduler>
    thread_id_type thread_pool<Scheduler>::set_state(
        util::steady_time_point const& abs_time,
        thread_id_type const& id, thread_state_enum newstate,
        thread_state_ex_enum newstate_ex, thread_priority priority,
        error_code& ec)
    {
        return detail::set_thread_state_timed(sched_, abs_time, id,
            newstate, newstate_ex, priority, get_worker_thread_num(), ec);
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::abort_all_suspended_threads()
    {
        sched_.Scheduler::abort_all_suspended_threads();
    }

    template <typename Scheduler>
    bool thread_pool<Scheduler>::cleanup_terminated(bool delete_all)
    {
        return sched_.Scheduler::cleanup_terminated(delete_all);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    std::size_t thread_pool<Scheduler>::get_worker_thread_num() const
    {
        return thread_num_tss_.get_worker_thread_num();
    }

    template <typename Scheduler>
    compat::thread& thread_pool<Scheduler>::get_os_thread_handle(
        std::size_t num_thread)
    {
        HPX_ASSERT(num_thread < threads_.size());
        return threads_[threads_.size() - num_thread - 1];
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::get_thread_count(
        thread_state_enum state, thread_priority priority,
        std::size_t num, bool reset) const
    {
        return sched_.Scheduler::get_thread_count(state, priority, num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::get_scheduler_utilization() const
    {
        return (std::accumulate(tasks_active_.begin(), tasks_active_.end(),
            std::int64_t(0)) * 100) / thread_count_.load();
    }

    template <typename Scheduler>
    bool thread_pool<Scheduler>::enumerate_threads(
        util::function_nonser<bool(thread_id_type)> const& f,
        thread_state_enum state) const
    {
        return sched_.Scheduler::enumerate_threads(f, state);
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::reset_thread_distribution()
    {
        return sched_.Scheduler::reset_thread_distribution();
    }

    template <typename Scheduler>
    void thread_pool<Scheduler>::set_scheduler_mode(
        threads::policies::scheduler_mode mode)
    {
        return sched_.set_scheduler_mode(mode);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    bool thread_pool<Scheduler>::run(std::unique_lock<compat::mutex>& l,
        std::size_t num_threads)
    {
        HPX_ASSERT(l.owns_lock());

        LTM_(info) //-V128
            << "thread_pool::run: " << pool_name_
            << " number of processing units available: " //-V128
            << threads::hardware_concurrency();
        LTM_(info) //-V128
            << "thread_pool::run: " << pool_name_
            << " creating " << num_threads << " OS thread(s)"; //-V128

        if (0 == num_threads) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "thread_pool::run", "number of threads is zero");
        }

        if (!threads_.empty() || sched_.has_reached_state(state_running))
            return true;    // do nothing if already running

        executed_threads_.resize(num_threads);
        executed_thread_phases_.resize(num_threads);

        tfunc_times_.resize(num_threads);
        exec_times_.resize(num_threads);

        idle_loop_counts_.resize(num_threads);
        busy_loop_counts_.resize(num_threads);

        reset_tfunc_times_.resize(num_threads);

        tasks_active_.resize(num_threads);

        // scale timestamps to nanoseconds
        std::uint64_t base_timestamp = util::hardware::timestamp();
        std::uint64_t base_time = util::high_resolution_clock::now();
        std::uint64_t curr_timestamp = util::hardware::timestamp();
        std::uint64_t curr_time = util::high_resolution_clock::now();

        while ((curr_time - base_time) <= 100000)
        {
            curr_timestamp = util::hardware::timestamp();
            curr_time = util::high_resolution_clock::now();
        }

        if (curr_timestamp - base_timestamp != 0)
        {
            timestamp_scale_ = double(curr_time - base_time) /
                double(curr_timestamp - base_timestamp);
        }

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
        // timestamps/values of last reset operation for various performance
        // counters
        reset_executed_threads_.resize(num_threads);
        reset_executed_thread_phases_.resize(num_threads);

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        // timestamps/values of last reset operation for various performance
        // counters
        reset_thread_duration_.resize(num_threads);
        reset_thread_duration_times_.resize(num_threads);

        reset_thread_overhead_.resize(num_threads);
        reset_thread_overhead_times_.resize(num_threads);
        reset_thread_overhead_times_total_.resize(num_threads);

        reset_thread_phase_duration_.resize(num_threads);
        reset_thread_phase_duration_times_.resize(num_threads);

        reset_thread_phase_overhead_.resize(num_threads);
        reset_thread_phase_overhead_times_.resize(num_threads);
        reset_thread_phase_overhead_times_total_.resize(num_threads);

        reset_cumulative_thread_duration_.resize(num_threads);

        reset_cumulative_thread_overhead_.resize(num_threads);
        reset_cumulative_thread_overhead_total_.resize(num_threads);
#endif
#endif

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
        reset_idle_rate_time_.resize(num_threads);
        reset_idle_rate_time_total_.resize(num_threads);

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
        reset_creation_idle_rate_time_.resize(num_threads);
        reset_creation_idle_rate_time_total_.resize(num_threads);

        reset_cleanup_idle_rate_time_.resize(num_threads);
        reset_cleanup_idle_rate_time_total_.resize(num_threads);
#endif
#endif

        LTM_(info)
            << "thread_pool::run: " << pool_name_
            << " timestamp_scale: " << timestamp_scale_; //-V128

        try {
            HPX_ASSERT(startup_.get() == nullptr);
            startup_.reset(
                new compat::barrier(static_cast<unsigned>(num_threads+1))
            );

            // run threads and wait for initialization to complete

            topology const& topology_ = get_topology();

            std::size_t thread_num = num_threads;
            while (thread_num-- != 0) {
                threads::mask_cref_type mask =
                    sched_.Scheduler::get_pu_mask(topology_, thread_num);

                LTM_(info) //-V128
                    << "thread_pool::run: " << pool_name_
                    << " create OS thread " << thread_num //-V128
                    << ": will run on processing units within this mask: "
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) || \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT <= 64)
                    << std::hex << "0x" << mask;
#else
                    << "0b" << mask;
#endif

                // create a new thread
                threads_.push_back(compat::thread(
                        &thread_pool::thread_func, this, thread_num,
                        std::ref(topology_), std::ref(*startup_)
                    ));

                // set the new threads affinity (on Windows systems)
                if (any(mask))
                {
                    error_code ec(lightweight);
                    topology_.set_thread_affinity_mask(threads_.back(), mask, ec);
                    if (ec)
                    {
                        LTM_(warning) //-V128
                            << "thread_pool::run: " << pool_name_
                            << " setting thread affinity on OS thread " //-V128
                            << thread_num << " failed with: "
                            << ec.get_message();
                    }
                }
                else
                {
                    LTM_(debug) //-V128
                        << "thread_pool::run: " << pool_name_
                        << " setting thread affinity on OS thread " //-V128
                        << thread_num << " was explicitly disabled.";
                }
            }

            // the main thread needs to have a unique thread_num
            init_tss(num_threads);
            startup_->wait();

            // The scheduler is now running.
            sched_.set_all_states(state_running);
        }
        catch (std::exception const& e) {
            LTM_(always)
                << "thread_pool::run: " << pool_name_
                << " failed with: " << e.what();

            // trigger the barrier
            if (startup_.get() != nullptr)
            {
                while (num_threads-- != 0)
                    startup_->wait();
            }

            stop(l);
            threads_.clear();

            return false;
        }

        LTM_(info) << "thread_pool::run: " << pool_name_ << " running";
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    void thread_pool<Scheduler>::stop (
        std::unique_lock<compat::mutex>& l, bool blocking)
    {
        HPX_ASSERT(l.owns_lock());

        return stop_locked(l, blocking);
    }

    template <typename Scheduler>
    template <typename Lock>
    void thread_pool<Scheduler>::stop_locked(Lock& l, bool blocking)
    {
        LTM_(info)
            << "thread_pool::stop: " << pool_name_
            << " blocking(" << std::boolalpha << blocking << ")";

        deinit_tss();

        if (!threads_.empty()) {
            // set state to stopping
            sched_.set_all_states(state_stopping);

            // make sure we're not waiting
            sched_.Scheduler::do_some_work(std::size_t(-1));

            if (blocking) {
                for (std::size_t i = 0; i != threads_.size(); ++i)
                {
                    // make sure no OS thread is waiting
                    LTM_(info)
                        << "thread_pool::stop: " << pool_name_
                        << " notify_all";

                    sched_.Scheduler::do_some_work(std::size_t(-1));

                    LTM_(info) //-V128
                        << "thread_pool::stop: " << pool_name_
                        << " join:" << i; //-V128

                    // unlock the lock while joining
                    util::unlock_guard<Lock> ul(l);
                    threads_[i].join();
                }
                threads_.clear();
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct manage_active_thread_count
    {
        manage_active_thread_count(boost::atomic<long>& counter)
          : counter_(counter)
        {
            ++counter_;
        }
        ~manage_active_thread_count()
        {
            --counter_;
        }

        boost::atomic<long>& counter_;
    };

    template <typename Scheduler>
    struct init_tss_helper
    {
        init_tss_helper(thread_pool<Scheduler>& pool, std::size_t thread_num)
          : pool_(pool), thread_num_(thread_num)
        {
            pool.notifier_.on_start_thread(thread_num);
            pool.init_tss(thread_num);
            pool.sched_.Scheduler::on_start_thread(thread_num);
        }
        ~init_tss_helper()
        {
            pool_.sched_.Scheduler::on_stop_thread(thread_num_);
            pool_.deinit_tss();
            pool_.notifier_.on_stop_thread(thread_num_);
        }

        thread_pool<Scheduler>& pool_;
        std::size_t thread_num_;
    };

    template <typename Scheduler>
    void thread_pool<Scheduler>::thread_func(std::size_t num_thread,
        topology const& topology, compat::barrier& startup)
    {
        // Set the affinity for the current thread.
        threads::mask_cref_type mask =
            sched_.Scheduler::get_pu_mask(topology, num_thread);

        if (LHPX_ENABLED(debug))
            topology.write_to_log();

        error_code ec(lightweight);
        if (any(mask))
        {
            topology.set_thread_affinity_mask(mask, ec);
            if (ec)
            {
                LTM_(warning) //-V128
                    << "thread_pool::thread_func: " << pool_name_
                    << " setting thread affinity on OS thread " //-V128
                    << num_thread << " failed with: " << ec.get_message();
            }
        }
        else
        {
            LTM_(debug) //-V128
                << "thread_pool::thread_func: " << pool_name_
                << " setting thread affinity on OS thread " //-V128
                << num_thread << " was explicitly disabled.";
        }

        // Setting priority of worker threads to a lower priority, this needs to
        // be done in order to give the parcel pool threads higher priority
        if ((mode_ & policies::reduce_thread_priority) &&
            any(mask & used_processing_units_))
        {
            topology.reduce_thread_priority(ec);
            if (ec)
            {
                LTM_(warning) //-V128
                    << "thread_pool::thread_func: " << pool_name_
                    << " reducing thread priority on OS thread " //-V128
                    << num_thread << " failed with: " << ec.get_message();
            }
        }

        // manage the number of this thread in its TSS
        init_tss_helper<Scheduler> tss_helper(*this, num_thread);

        // wait for all threads to start up before before starting HPX work
        startup.wait();

        {
            LTM_(info) //-V128
                << "thread_pool::thread_func: " << pool_name_
                << " starting OS thread: " << num_thread; //-V128

            try {
                try {
                    manage_active_thread_count count(thread_count_);

                    // run the work queue
                    hpx::threads::coroutines::prepare_main_thread main_thread;

                    // run main Scheduler loop until terminated
                    detail::scheduling_counters counters(
                        executed_threads_[num_thread],
                        executed_thread_phases_[num_thread],
                        tfunc_times_[num_thread], exec_times_[num_thread],
                        idle_loop_counts_[num_thread], busy_loop_counts_[num_thread],
                        tasks_active_[num_thread]);

                    detail::scheduling_callbacks callbacks(
                        util::bind( //-V107
                            &policies::scheduler_base::idle_callback,
                            &sched_, num_thread
                        ),
                        detail::scheduling_callbacks::callback_type());

                    if (mode_ & policies::do_background_work)
                    {
                        callbacks.background_ = util::bind( //-V107
                            &policies::scheduler_base::background_callback,
                            &sched_, num_thread);
                    }

                    sched_.set_scheduler_mode(mode_);
                    detail::scheduling_loop(num_thread, sched_, counters,
                        callbacks);

                    // the OS thread is allowed to exit only if no more HPX
                    // threads exist or if some other thread has terminated
                    HPX_ASSERT(!sched_.Scheduler::get_thread_count(
                            unknown, thread_priority_default, num_thread) ||
                        sched_.get_state(num_thread) == state_terminating);
                }
                catch (hpx::exception const& e) {
                    LFATAL_ //-V128
                        << "thread_pool::thread_func: " << pool_name_
                        << " thread_num:" << num_thread //-V128
                        << " : caught hpx::exception: "
                        << e.what() << ", aborted thread execution";

                    report_error(num_thread, boost::current_exception());
                    return;
                }
                catch (boost::system::system_error const& e) {
                    LFATAL_ //-V128
                        << "thread_pool::thread_func: " << pool_name_
                        << " thread_num:" << num_thread //-V128
                        << " : caught boost::system::system_error: "
                        << e.what() << ", aborted thread execution";

                    report_error(num_thread, boost::current_exception());
                    return;
                }
                catch (std::exception const& e) {
                    // Repackage exceptions to avoid slicing.
                    boost::throw_exception(boost::enable_error_info(
                        hpx::exception(unhandled_exception, e.what())));
                }
            }
            catch (...) {
                LFATAL_ //-V128
                    << "thread_pool::thread_func: " << pool_name_
                    << " thread_num:" << num_thread //-V128
                    << " : caught unexpected " //-V128
                       "exception, aborted thread execution";

                report_error(num_thread, boost::current_exception());
                return;
            }

            LTM_(info) //-V128
                << "thread_pool::thread_func: " << pool_name_
                << " thread_num: " << num_thread
                << " : ending OS thread, " //-V128
                   "executed " << executed_threads_[num_thread]
                << " HPX threads";
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // performance counters
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS)
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_cumulative_duration(std::size_t num, bool reset)
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
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::avg_idle_rate(bool reset)
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

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::avg_idle_rate(
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

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::avg_creation_idle_rate(bool reset)
    {
        double const creation_total =
            static_cast<double>(sched_.get_creation_time(reset));

        std::uint64_t exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total = std::accumulate(
            reset_creation_idle_rate_time_.begin(),
            reset_creation_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total = std::accumulate(
            reset_creation_idle_rate_time_total_.begin(),
            reset_creation_idle_rate_time_total_.end(), std::uint64_t(0));

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

        if (tfunc_total == exec_total)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent = (creation_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::avg_cleanup_idle_rate(bool reset)
    {
        double const cleanup_total =
            static_cast<double>(sched_.get_cleanup_time(reset));

        std::uint64_t exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), std::uint64_t(0));
        std::uint64_t tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), std::uint64_t(0));
        std::uint64_t reset_exec_total = std::accumulate(
            reset_cleanup_idle_rate_time_.begin(),
            reset_cleanup_idle_rate_time_.end(), std::uint64_t(0));
        std::uint64_t reset_tfunc_total = std::accumulate(
            reset_cleanup_idle_rate_time_total_.begin(),
            reset_cleanup_idle_rate_time_total_.end(), std::uint64_t(0));

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

        if (tfunc_total == exec_total)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent = (cleanup_total / double(tfunc_total - exec_total));
        return std::int64_t(10000. * percent);    // 0.01 percent
    }
#endif
#endif

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_queue_length(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_queue_length(num_thread);
    }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_average_thread_wait_time(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_average_thread_wait_time(num_thread);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_average_task_wait_time(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_average_task_wait_time(num_thread);
    }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_pending_misses(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_pending_misses(num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_pending_accesses(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_pending_accesses(num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_stolen_from_pending(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_from_pending(num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_stolen_to_pending(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_to_pending(num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_stolen_from_staged(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_from_staged(num, reset);
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::
        get_num_stolen_to_staged(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_to_staged(num, reset);
    }
#endif

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::get_idle_loop_count(std::size_t num) const
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(idle_loop_counts_.begin(),
                idle_loop_counts_.end(), 0ll);
        }
        return idle_loop_counts_[num];
    }

    template <typename Scheduler>
    std::int64_t thread_pool<Scheduler>::get_busy_loop_count(std::size_t num) const
    {
        if (num == std::size_t(-1))
        {
            return std::accumulate(busy_loop_counts_.begin(),
                busy_loop_counts_.end(), 0ll);
        }
        return busy_loop_counts_[num];
    }
}}}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
#if defined(HPX_HAVE_THROTTLE_SCHEDULER) && defined(HPX_HAVE_APEX)
#include <hpx/runtime/threads/policies/throttle_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::throttle_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_LOCAL_SCHEDULER)
#include <hpx/runtime/threads/policies/local_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::local_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_STATIC_SCHEDULER)
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::static_queue_scheduler<> >;
#endif

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/static_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::static_priority_queue_scheduler<> >;
#endif

#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<
        hpx::compat::mutex, hpx::threads::policies::lockfree_fifo
    > >;
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<
        hpx::compat::mutex, hpx::threads::policies::lockfree_lifo
    > >;

#if defined(HPX_HAVE_ABP_SCHEDULER)
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::local_priority_queue_scheduler<
        hpx::compat::mutex, hpx::threads::policies::lockfree_abp_fifo
    > >;
#endif

#if defined(HPX_HAVE_HIERARCHY_SCHEDULER)
#include <hpx/runtime/threads/policies/hierarchy_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::hierarchy_scheduler<> >;
#endif

#if defined(HPX_HAVE_PERIODIC_PRIORITY_SCHEDULER)
#include <hpx/runtime/threads/policies/periodic_priority_queue_scheduler.hpp>
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::periodic_priority_queue_scheduler<> >;
#endif

