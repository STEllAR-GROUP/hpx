//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/util/unlock_guard.hpp>

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS) && \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
#include <hpx/util/hardware/timestamp.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#endif

#include <boost/ref.hpp>
#include <boost/exception_ptr.hpp>

#include <cstdint>
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
#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS) && \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
        timestamp_scale_ = 1.0;
#endif
    }

    template <typename Scheduler>
    thread_pool<Scheduler>::~thread_pool()
    {
        if (!threads_.empty()) {
            if (!sched_.has_reached_state(state_suspended))
            {
                // still running
                lcos::local::no_mutex mtx;
                boost::unique_lock<lcos::local::no_mutex> l(mtx);
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
        std::size_t num_thread = get_worker_thread_num() % thread_count_;
        if (num_thread != std::size_t(-1))
            return get_state(num_thread);
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
    boost::thread& thread_pool<Scheduler>::get_os_thread_handle(
        std::size_t num_thread)
    {
        HPX_ASSERT(num_thread < threads_.size());
        return threads_[threads_.size() - num_thread - 1];
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::get_thread_count(
        thread_state_enum state, thread_priority priority,
        std::size_t num, bool reset) const
    {
        return sched_.Scheduler::get_thread_count(state, priority, num, reset);
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
    bool thread_pool<Scheduler>::run(boost::unique_lock<boost::mutex>& l,
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

#if defined(HPX_HAVE_THREAD_CUMULATIVE_COUNTS) && \
    defined(HPX_HAVE_THREAD_IDLE_RATES)
        // scale timestamps to nanoseconds
        boost::uint64_t base_timestamp = util::hardware::timestamp();
        boost::uint64_t base_time = util::high_resolution_clock::now();
        boost::uint64_t curr_timestamp = util::hardware::timestamp();
        boost::uint64_t curr_time = util::high_resolution_clock::now();

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

        LTM_(info)
            << "thread_pool::run: " << pool_name_
            << " timestamp_scale: " << timestamp_scale_; //-V128
#endif

        if (!threads_.empty() || sched_.has_reached_state(state_running))
            return true;    // do nothing if already running

        executed_threads_.resize(num_threads);
        executed_thread_phases_.resize(num_threads);
        tfunc_times_.resize(num_threads);
        exec_times_.resize(num_threads);

        try {
            HPX_ASSERT(startup_.get() == 0);
            startup_.reset(
                new boost::barrier(static_cast<unsigned>(num_threads+1))
            );

            // run threads and wait for initialization to complete
            sched_.set_all_states(state_running);

            topology const& topology_ = get_topology();

            std::size_t thread_num = num_threads;
            while (thread_num-- != 0) {
                threads::mask_cref_type mask =
                    sched_.Scheduler::get_pu_mask(topology_, thread_num);

                LTM_(info) //-V128
                    << "thread_pool::run: " << pool_name_
                    << " create OS thread " << thread_num //-V128
                    << ": will run on processing units within this mask: "
#if !defined(HPX_WITH_MORE_THAN_64_THREADS) || \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT <= 64)
                    << std::hex << "0x" << mask;
#else
                    << "0b" << mask;
#endif

                // create a new thread
                threads_.push_back(new boost::thread(
                        util::bind(&thread_pool::thread_func, this, thread_num,
                            boost::ref(topology_), boost::ref(*startup_))
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
        }
        catch (std::exception const& e) {
            LTM_(always)
                << "thread_pool::run: " << pool_name_
                << " failed with: " << e.what();

            // trigger the barrier
            if (startup_.get() != 0)
            {
                while (num_threads-- != 0 && !startup_->wait())
                    ;
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
        boost::unique_lock<boost::mutex>& l, bool blocking)
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
        topology const& topology, boost::barrier& startup)
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
                    hpx::util::coroutines::prepare_main_thread main_thread;

                    // run main Scheduler loop until terminated
                    detail::scheduling_counters counters(
                        executed_threads_[num_thread],
                        executed_thread_phases_[num_thread],
                        tfunc_times_[num_thread], exec_times_[num_thread]);

                    detail::scheduling_callbacks callbacks(
                        util::bind(
                            &policies::scheduler_base::idle_callback,
                            &sched_, num_thread
                        ),
                        detail::scheduling_callbacks::callback_type());

                    if (mode_ & policies::do_background_work)
                    {
                        callbacks.background_ = util::bind(
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
#ifdef HPX_HAVE_THREAD_CUMULATIVE_COUNTS
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_executed_threads(std::size_t num, bool reset)
    {
        boost::int64_t result = 0;
        if (num != std::size_t(-1)) {
            result = executed_threads_[num];
            if (reset)
                executed_threads_[num] = 0;
            return result;
        }

        result = std::accumulate(executed_threads_.begin(),
            executed_threads_.end(), 0LL);
        if (reset)
            std::fill(executed_threads_.begin(), executed_threads_.end(), 0LL);
        return result;
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_executed_thread_phases(std::size_t num, bool reset)
    {
        boost::int64_t result = 0;
        if (num != std::size_t(-1)) {
            result = executed_thread_phases_[num];
            if (reset)
                executed_thread_phases_[num] = 0;
            return result;
        }

        result = std::accumulate(executed_thread_phases_.begin(),
            executed_thread_phases_.end(), 0LL);
        if (reset) {
            std::fill(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), 0LL);
        }
        return result;
    }

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_thread_phase_duration(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);
            double num_phases = static_cast<double>(executed_thread_phases_[num]);

            if (reset) {
                executed_thread_phases_[num] = 0;
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t((exec_total * timestamp_scale_)/ num_phases);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);
        double num_phases = std::accumulate(executed_thread_phases_.begin(),
            executed_thread_phases_.end(), 0.);

        if (reset) {
            std::fill(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), 0LL);
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t((exec_total * timestamp_scale_)/ num_phases);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_thread_duration(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);
            double num_threads = static_cast<double>(executed_threads_[num]);

            if (reset) {
                executed_threads_[num] = 0;
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t((exec_total * timestamp_scale_)/ num_threads);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);
        double num_threads = std::accumulate(executed_threads_.begin(),
            executed_threads_.end(), 0.);

        if (reset) {
            std::fill(executed_threads_.begin(), executed_threads_.end(), 0LL);
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t((exec_total * timestamp_scale_) / num_threads);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_thread_phase_overhead(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);
            double tfunc_total = static_cast<double>(tfunc_times_[num]);
            double num_phases = static_cast<double>(executed_thread_phases_[num]);

            if (reset) {
                executed_thread_phases_[num] = 0;
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t(((tfunc_total - exec_total) * timestamp_scale_)/
                    num_phases);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);
        double tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), 0.);
        double num_phases = std::accumulate(executed_thread_phases_.begin(),
            executed_thread_phases_.end(), 0.);

        if (reset) {
            std::fill(executed_thread_phases_.begin(),
                executed_thread_phases_.end(), 0LL);
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t(((tfunc_total - exec_total) * timestamp_scale_)/
                num_phases);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_thread_overhead(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);
            double tfunc_total = static_cast<double>(tfunc_times_[num]);
            double num_threads = static_cast<double>(executed_threads_[num]);

            if (reset) {
                executed_threads_[num] = 0;
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t(((tfunc_total - exec_total) *
                        timestamp_scale_) / num_threads);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);
        double tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), 0.);
        double num_threads = std::accumulate(executed_threads_.begin(),
            executed_threads_.end(), 0.);

        if (reset) {
            std::fill(executed_threads_.begin(), executed_threads_.end(), 0LL);
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t(((tfunc_total - exec_total) *
                        timestamp_scale_) / num_threads);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_cumulative_thread_duration(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);

            if (reset) {
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t(exec_total * timestamp_scale_);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);

        if (reset) {
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t(exec_total * timestamp_scale_);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_cumulative_thread_overhead(std::size_t num, bool reset)
    {
        if (num != std::size_t(-1)) {
            double exec_total = static_cast<double>(exec_times_[num]);
            double tfunc_total = static_cast<double>(tfunc_times_[num]);

            if (reset) {
                tfunc_times_[num] = boost::uint64_t(-1);
            }
            return boost::uint64_t((tfunc_total - exec_total) * timestamp_scale_);
        }

        double exec_total = std::accumulate(exec_times_.begin(),
            exec_times_.end(), 0.);
        double tfunc_total = std::accumulate(tfunc_times_.begin(),
            tfunc_times_.end(), 0.);

        if (reset) {
            std::fill(executed_threads_.begin(), executed_threads_.end(), 0LL);
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }
        return boost::uint64_t((tfunc_total - exec_total) * timestamp_scale_);
    }
#endif
#endif

#ifdef HPX_HAVE_THREAD_IDLE_RATES
    ///////////////////////////////////////////////////////////////////////////
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::avg_idle_rate(bool reset)
    {
        double const exec_total =
            std::accumulate(exec_times_.begin(), exec_times_.end(), 0.);
        double const tfunc_total =
            std::accumulate(tfunc_times_.begin(), tfunc_times_.end(), 0.);

        if (reset) {
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }

        if (std::abs(tfunc_total) < 1e-16)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_total >= exec_total);

        double const percent = 1. - (exec_total / tfunc_total);
        return boost::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::avg_idle_rate(
        std::size_t num_thread, bool reset)
    {
        double const exec_time = static_cast<double>(exec_times_[num_thread]);
        double const tfunc_time = static_cast<double>(tfunc_times_[num_thread]);

        if (reset) {
            tfunc_times_[num_thread] = boost::uint64_t(-1);
        }

        if (std::abs(tfunc_time) < 1e-16)   // avoid division by zero
            return 10000LL;

        HPX_ASSERT(tfunc_time > exec_time);

        double const percent = 1. - (exec_time / tfunc_time);
        return boost::int64_t(10000. * percent);   // 0.01 percent
    }

#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::avg_creation_idle_rate(bool reset)
    {
        double const creation_total =
            static_cast<double>(sched_.get_creation_time(reset));
        double const exec_total =
            std::accumulate(exec_times_.begin(), exec_times_.end(), 0.);
        double const tfunc_total =
            std::accumulate(tfunc_times_.begin(), tfunc_times_.end(), 0.);

        if (reset) {
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }

        // avoid division by zero
        if (std::abs(tfunc_total - exec_total) < 1e-16)
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent = (creation_total / (tfunc_total - exec_total));
        return boost::int64_t(10000. * percent);    // 0.01 percent
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::avg_cleanup_idle_rate(bool reset)
    {
        double const cleanup_total =
            static_cast<double>(sched_.get_cleanup_time(reset));
        double const exec_total =
            std::accumulate(exec_times_.begin(), exec_times_.end(), 0.);
        double const tfunc_total =
            std::accumulate(tfunc_times_.begin(), tfunc_times_.end(), 0.);

        if (reset) {
            std::fill(tfunc_times_.begin(), tfunc_times_.end(),
                boost::uint64_t(-1));
        }

        // avoid division by zero
        if (std::abs(tfunc_total - exec_total) < 1e-16)
            return 10000LL;

        HPX_ASSERT(tfunc_total > exec_total);

        double const percent = (cleanup_total / (tfunc_total - exec_total));
        return boost::int64_t(10000. * percent);    // 0.01 percent
    }
#endif
#endif

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_queue_length(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_queue_length(num_thread);
    }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_average_thread_wait_time(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_average_thread_wait_time(num_thread);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_average_task_wait_time(std::size_t num_thread) const
    {
        return sched_.Scheduler::get_average_task_wait_time(num_thread);
    }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_pending_misses(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_pending_misses(num, reset);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_pending_accesses(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_pending_accesses(num, reset);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_stolen_from_pending(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_from_pending(num, reset);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_stolen_to_pending(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_to_pending(num, reset);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_stolen_from_staged(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_from_staged(num, reset);
    }

    template <typename Scheduler>
    boost::int64_t thread_pool<Scheduler>::
        get_num_stolen_to_staged(std::size_t num, bool reset)
    {
        return sched_.Scheduler::get_num_stolen_to_staged(num, reset);
    }
#endif

}}}

///////////////////////////////////////////////////////////////////////////////
/// explicit template instantiation for the thread manager of our choice
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
    hpx::threads::policies::local_priority_queue_scheduler<> >;

#if defined(HPX_HAVE_ABP_SCHEDULER)
template class HPX_EXPORT hpx::threads::detail::thread_pool<
    hpx::threads::policies::abp_fifo_priority_queue_scheduler>;
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

