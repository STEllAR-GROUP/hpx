//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREAD_POOL_HPP)
#define HPX_THREAD_POOL_HPP

#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
#include <hpx/runtime/threads/detail/scheduling_loop.hpp>
#include <hpx/runtime/threads/detail/set_thread_state.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/schedulers.hpp>
#include <hpx/runtime/threads/threadmanager_impl.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/unlock_guard.hpp>

namespace hpx { namespace threads
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        struct init_tss_helper;

        ///////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        class thread_pool_impl : public thread_pool
        {
        public:
            ///////////////////////////////////////////////////////////////////
            thread_pool_impl(std::unique_ptr<Scheduler> sched,
                threads::policies::callback_notifier &notifier,
                std::size_t index, char const *pool_name,
                policies::scheduler_mode m =
                    policies::scheduler_mode::nothing_special,
                std::size_t thread_offset = 0)
              : thread_pool(notifier, index, pool_name, m)
              , sched_(std::move(sched))
              , thread_offset_(thread_offset)
            {
                timestamp_scale_ = 1.0;
                sched_->set_parent_pool(this);
            }

            virtual ~thread_pool_impl()
            {
                if (!threads_.empty())
                {
                    if (!sched_->has_reached_state(state_suspended))
                    {
                        // still running
                        lcos::local::no_mutex mtx;
                        std::unique_lock<lcos::local::no_mutex> l(mtx);
                        stop_locked(l);
                    }
                    threads_.clear();
                }
            }

            void print_pool()
            {
                std::cout << "[pool \"" << id_.name_ << "\", #" << id_.index_
                          << "] with scheduler " << sched_->get_scheduler_name()
                          << "\n"
                          << "is running on PUs : \n";
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) || \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT <= 64)
                std::cout << std::hex << "0x" << used_processing_units_ << '\n';
#else
                std::cout << "0b" << used_processing_units_ << '\n';
#endif
            }

            threads::policies::scheduler_base *get_scheduler() const
            {
                return sched_.get();
            }

            ///////////////////////////////////////////////////////////////////

            hpx::state get_state() const
            {
                // get_worker_thread_num returns the global thread number which
                // might
                // be too large. This function might get called from within
                // background_work inside the os executors
                if (thread_count_ != 0)
                {
                    std::size_t num_thread =
                        get_worker_thread_num() % thread_count_;
                    if (num_thread != std::size_t(-1))
                        return get_state(num_thread);
                }
                return sched_->get_minmax_state().second;
            }

            hpx::state get_state(std::size_t num_thread) const
            {
                HPX_ASSERT(num_thread != std::size_t(-1));
                return sched_->get_state(num_thread).load();
            }

            bool has_reached_state(hpx::state s) const
            {
                return sched_->has_reached_state(s);
            }

            ///////////////////////////////////////////////////////////////////
            void init(std::size_t pool_threads, std::size_t threads_offset)
            {
                resize(used_processing_units_, threads::hardware_concurrency());
                for (std::size_t i = 0; i != pool_threads; ++i)
                    used_processing_units_ |=
                        get_resource_partitioner().get_pu_mask(
                            threads_offset + i, sched_->numa_sensitive());
            }

            ///////////////////////////////////////////////////////////////////
            void do_some_work(std::size_t num_thread)
            {
                sched_->Scheduler::do_some_work(num_thread);
            }

            inline std::size_t get_thread_offset() const
            {
                return thread_offset_;
            }

            void report_error(std::size_t num, std::exception_ptr const &e)
            {
                sched_->set_all_states(state_terminating);
                notifier_.on_error(num, e);
                sched_->Scheduler::on_error(num, e);
            }

            void create_thread(thread_init_data &data, thread_id_type &id,
                thread_state_enum initial_state, bool run_now, error_code &ec)
            {
                // verify state
                if (thread_count_ == 0 && !sched_->is_state(state_running))
                {
                    // thread-manager is not currently running
                    HPX_THROWS_IF(ec, invalid_status,
                        "thread_pool<Scheduler>::create_thread",
                        "invalid state: thread pool is not running");
                    return;
                }

                detail::create_thread(sched_.get(), data, id, initial_state, 
                    run_now, ec);    //-V601
            }

            void create_work(thread_init_data &data,
                thread_state_enum initial_state, error_code &ec)
            {
                // verify state
                if (thread_count_ == 0 && !sched_->is_state(state_running))
                {
                    // thread-manager is not currently running
                    HPX_THROWS_IF(ec, invalid_status,
                        "thread_pool<Scheduler>::create_work",
                        "invalid state: thread pool is not running");
                    return;
                }

                detail::create_work(sched_.get(), data, initial_state, ec);    //-V601
            }

            thread_id_type set_state(util::steady_time_point const &abs_time,
                thread_id_type const &id, thread_state_enum newstate,
                thread_state_ex_enum newstate_ex, thread_priority priority,
                error_code &ec)
            {
                return detail::set_thread_state_timed(*sched_, abs_time, id,
                    newstate, newstate_ex, priority, get_worker_thread_num(),
                    ec);
            }

            void abort_all_suspended_threads()
            {
                sched_->Scheduler::abort_all_suspended_threads();
            }

            bool cleanup_terminated(bool delete_all)
            {
                return sched_->Scheduler::cleanup_terminated(delete_all);
            }

            std::int64_t get_thread_count(thread_state_enum state,
                thread_priority priority, std::size_t num, bool reset) const
            {
                return sched_->Scheduler::get_thread_count(
                    state, priority, num, reset);
            }

            bool enumerate_threads(
                util::function_nonser<bool(thread_id_type)> const &f,
                thread_state_enum state) const
            {
                return sched_->Scheduler::enumerate_threads(f, state);
            }

            void reset_thread_distribution()
            {
                return sched_->Scheduler::reset_thread_distribution();
            }

            void set_scheduler_mode(threads::policies::scheduler_mode mode)
            {
                return sched_->set_scheduler_mode(mode);
            }

            ///////////////////////////////////////////////////////////////////
            bool run(std::unique_lock<compat::mutex> &l,
                compat::barrier &startup, std::size_t pool_threads)
            {
                HPX_ASSERT(l.owns_lock());

                LTM_(info)    //-V128
                    << "thread_pool::run: " << id_.name_
                    << " number of processing units available: "    //-V128
                    << threads::hardware_concurrency();
                LTM_(info)    //-V128
                    << "thread_pool::run: " << id_.name_ << " creating "
                    << pool_threads << " OS thread(s)";    //-V128

                if (0 == pool_threads)
                {
                    HPX_THROW_EXCEPTION(bad_parameter, "thread_pool::run",
                        "number of threads is zero");
                }

                if (!threads_.empty() || sched_->has_reached_state(state_running))
                    return true;    // do nothing if already running

                executed_threads_.resize(pool_threads);
                executed_thread_phases_.resize(pool_threads);

                tfunc_times_.resize(pool_threads);
                exec_times_.resize(pool_threads);

                idle_loop_counts_.resize(pool_threads);
                busy_loop_counts_.resize(pool_threads);

                reset_tfunc_times_.resize(pool_threads);

                tasks_active_.resize(pool_threads);

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
                // timestamps/values of last reset operation for various
                // performance
                // counters
                reset_executed_threads_.resize(pool_threads);
                reset_executed_thread_phases_.resize(pool_threads);

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
                // timestamps/values of last reset operation for various
                // performance
                // counters
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

                LTM_(info) << "thread_pool::run: " << id_.name_
                           << " timestamp_scale: "
                           << timestamp_scale_;    //-V128

                //! TODO add try ... catch
                //            try {
                // run threads and wait for initialization to complete

                topology const &topology_ = get_topology();

                for (std::size_t thread_num_(0); thread_num_ < pool_threads;
                     thread_num_++)
                {
                    std::size_t global_thread_num =
                        thread_offset_ + thread_num_;
                    threads::mask_cref_type mask =
                        get_resource_partitioner().get_pu_mask(
                            global_thread_num, sched_->numa_sensitive());

                    // thread_num ordering: 1. threads of default pool
                    //                      2. threads of first special pool
                    //                      3. etc.
                    // get_pu_mask expects index according to ordering of masks
                    // in affinity_data::affinity_masks_
                    // which is in order of occupied PU
                    LTM_(info)    //-V128
                        << "thread_pool::run: " << id_.name_
                        << " create OS thread "
                        << global_thread_num    //-V128 //! BOTH?
                        << ": will run on processing units within this "
                           "mask: "
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) ||                                 \
    (defined(HPX_HAVE_MAX_CPU_COUNT) && HPX_HAVE_MAX_CPU_COUNT <= 64)
                        << std::hex << "0x" << mask;
#else
                        << "0b" << mask;
#endif

                    // create a new thread
                    threads_.push_back(compat::thread(&thread_pool::thread_func,
                        this, global_thread_num, std::ref(topology_),
                        std::ref(startup)));

                    // set the new threads affinity (on Windows systems)
                    if (any(mask))
                    {
                        error_code ec(lightweight);
                        topology_.set_thread_affinity_mask(
                            threads_.back(), mask, ec);
                        if (ec)
                        {
                            LTM_(warning)    //-V128
                                << "thread_pool::run: " << id_.name_
                                << " setting thread affinity "
                                   "on OS thread "    //-V128
                                << pool_threads
                                << " failed with: " << ec.get_message();
                        }
                    }
                    else
                    {
                        LTM_(debug)    //-V128
                            << "thread_pool::run: " << id_.name_
                            << " setting thread affinity on "
                               "OS thread "    //-V128
                            << pool_threads << " was explicitly disabled.";
                    }
                }

                //! TODO add try ... catch
                /*            }
            catch (std::exception const &e) {
                LTM_(always) << "thread_pool::run: " << id_.name_
                             << " failed with: " << e.what();

                // trigger the barrier
                if (startup_.get() != nullptr) {
                    while (pool_threads-- != 0)
                        startup_->wait();
                }

                stop(l);
                threads_.clear();

                return false;
            }*/

                LTM_(info) << "thread_pool::run: " << id_.name_ << " running";
                return true;
            }

            ///////////////////////////////////////////////////////////////////
            bool run(
                std::unique_lock<compat::mutex> &l, std::size_t pool_threads)
            {
                compat::barrier startup(pool_threads + 1);
                bool ret = run(l, startup, pool_threads);
                startup.wait();
                return ret;
            }

            void stop_locked(std::unique_lock<lcos::local::no_mutex> &l,
                bool blocking = true)
            {
                LTM_(info) << "thread_pool::stop: " << id_.name_ << " blocking("
                           << std::boolalpha << blocking << ")";

                threads::get_thread_manager().deinit_tss();

                if (!threads_.empty())
                {
                    // set state to stopping
                    sched_->set_all_states(state_stopping);

                    // make sure we're not waiting
                    sched_->Scheduler::do_some_work(std::size_t(-1));

                    if (blocking)
                    {
                        for (std::size_t i = 0; i != threads_.size(); ++i)
                        {
                            // make sure no OS thread is waiting
                            LTM_(info) << "thread_pool::stop: " << id_.name_
                                       << " notify_all";

                            sched_->Scheduler::do_some_work(std::size_t(-1));

                            LTM_(info)    //-V128
                                << "thread_pool::stop: " << id_.name_
                                << " join:" << i;    //-V128

                            // unlock the lock while joining
                            util::unlock_guard<
                                std::unique_lock<lcos::local::no_mutex>>
                                ul(l);
                            threads_[i].join();
                        }
                        threads_.clear();
                    }
                }
            }

            void stop_locked(
                std::unique_lock<compat::mutex> &l, bool blocking = true)
            {
                LTM_(info) << "thread_pool::stop: " << id_.name_ << " blocking("
                           << std::boolalpha << blocking << ")";

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
                            LTM_(info) << "thread_pool::stop: " << id_.name_
                                       << " notify_all";

                            sched_->Scheduler::do_some_work(std::size_t(-1));

                            LTM_(info)    //-V128
                                << "thread_pool::stop: " << id_.name_
                                << " join:" << i;    //-V128

                            // unlock the lock while joining
                            util::unlock_guard<std::unique_lock<compat::mutex>>
                                ul(l);
                            threads_[i].join();
                        }
                        threads_.clear();
                    }
                }
            }

            ///////////////////////////////////////////////////////////////////
            compat::thread &get_os_thread_handle(std::size_t global_thread_num)
            {
                std::size_t num_thread_local =
                    global_thread_num - thread_offset_;
                HPX_ASSERT(num_thread_local < threads_.size());
                return threads_[num_thread_local];
            }

            void thread_func(std::size_t global_thread_num,
                topology const &topology, compat::barrier &startup)
            {
                // Set the affinity for the current thread.
                threads::mask_cref_type mask =
                    get_resource_partitioner().get_pu_mask(
                        global_thread_num, sched_->numa_sensitive());

                if (LHPX_ENABLED(debug))
                    topology.write_to_log();

                error_code ec(lightweight);
                if (any(mask))
                {
                    topology.set_thread_affinity_mask(mask, ec);
                    if (ec)
                    {
                        LTM_(warning)    //-V128
                            << "thread_pool::thread_func: " << id_.name_
                            << " setting thread affinity on OS "
                               "thread "    //-V128
                            << global_thread_num
                            << " failed with: " << ec.get_message();
                    }
                }
                else
                {
                    LTM_(debug)    //-V128
                        << "thread_pool::thread_func: " << id_.name_
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
                            << "thread_pool::thread_func: " << id_.name_
                            << " reducing thread priority on OS "
                               "thread "    //-V128
                            << global_thread_num
                            << " failed with: " << ec.get_message();
                    }
                }

                // manage the number of this thread in its TSS
                init_tss_helper<Scheduler> tss_helper(*this, global_thread_num);

                // wait for all threads to start up before before starting HPX
                // work
                startup.wait();

                {
                    LTM_(info)    //-V128
                        << "thread_pool::thread_func: " << id_.name_
                        << " starting OS thread: "
                        << global_thread_num;    //-V128

                    try
                    {
                        try
                        {
                            manage_active_thread_count count(thread_count_);

                            // run the work queue
                            hpx::threads::coroutines::prepare_main_thread
                                main_thread;

                            // run main Scheduler loop until terminated
                            detail::scheduling_counters counters(
                                executed_threads_[global_thread_num],
                                executed_thread_phases_[global_thread_num],
                                tfunc_times_[global_thread_num],
                                exec_times_[global_thread_num],
                                idle_loop_counts_[global_thread_num],
                                busy_loop_counts_[global_thread_num],
                                tasks_active_[global_thread_num]);

                            detail::scheduling_callbacks callbacks(
                                util::bind(    //-V107
                                    &policies::scheduler_base::idle_callback,
                                    std::ref(sched_), global_thread_num),
                                detail::scheduling_callbacks::callback_type());

                            if (mode_ & policies::do_background_work)
                            {
                                callbacks.background_ = util::bind(    //-V107
                                    &policies::scheduler_base::
                                        background_callback,
                                    std::ref(sched_), global_thread_num);
                            }

                            sched_->set_scheduler_mode(mode_);
                            detail::scheduling_loop(
                                global_thread_num - thread_offset_, *sched_,
                                counters, callbacks);

                            // the OS thread is allowed to exit only if no more HPX
                            // threads exist or if some other thread has terminated
                            HPX_ASSERT(
                                !sched_->Scheduler::get_thread_count(unknown,
                                    thread_priority_default,
                                    global_thread_num - thread_offset_) ||
                                sched_->get_state(global_thread_num -
                                    thread_offset_) == state_terminating);
                        }
                        catch (hpx::exception const &e)
                        {
                            LFATAL_    //-V128
                                << "thread_pool::thread_func: " << id_.name_
                                << " thread_num:"
                                << global_thread_num    //-V128
                                << " : caught hpx::exception: " << e.what()
                                << ", aborted thread execution";

                            report_error(
                                global_thread_num, std::current_exception());
                            return;
                        }
                        catch (boost::system::system_error const &e)
                        {
                            LFATAL_    //-V128
                                << "thread_pool::thread_func: " << id_.name_
                                << " thread_num:"
                                << global_thread_num    //-V128
                                << " : caught boost::system::system_error: "
                                << e.what() << ", aborted thread execution";

                            report_error(
                                global_thread_num, std::current_exception());
                            return;
                        }
                        catch (std::exception const &e)
                        {
                            // Repackage exceptions to avoid slicing.
                            boost::throw_exception(boost::enable_error_info(
                                hpx::exception(unhandled_exception, e.what())));
                        }
                    }
                    catch (...)
                    {
                        LFATAL_    //-V128
                            << "thread_pool::thread_func: " << id_.name_
                            << " thread_num:" << global_thread_num    //-V128
                            << " : caught unexpected "                //-V128
                               "exception, aborted thread execution";

                        report_error(
                            global_thread_num, std::current_exception());
                        return;
                    }

                    LTM_(info)    //-V128
                        << "thread_pool::thread_func: " << id_.name_
                        << " thread_num: " << global_thread_num
                        << " : ending OS thread, "    //-V128
                           "executed "
                        << executed_threads_[global_thread_num]
                        << " HPX threads";
                }
            }

#if defined(HPX_HAVE_THREAD_IDLE_RATES)
#if defined(HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES)
            std::int64_t avg_creation_idle_rate(bool reset)
            {
                double const creation_total =
                    static_cast<double>(sched_->get_creation_time(reset));

                std::uint64_t exec_total = std::accumulate(
                    exec_times_.begin(), exec_times_.end(), std::uint64_t(0));
                std::uint64_t tfunc_total = std::accumulate(
                    tfunc_times_.begin(), tfunc_times_.end(), std::uint64_t(0));
                std::uint64_t reset_exec_total =
                    std::accumulate(reset_creation_idle_rate_time_.begin(),
                        reset_creation_idle_rate_time_.end(), std::uint64_t(0));
                std::uint64_t reset_tfunc_total = std::accumulate(
                    reset_creation_idle_rate_time_total_.begin(),
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

            std::int64_t avg_cleanup_idle_rate(bool reset)
            {
                double const cleanup_total =
                    static_cast<double>(sched_->get_cleanup_time(reset));

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
#endif
            std::int64_t get_queue_length(std::size_t num_thread) const
            {
                return sched_->Scheduler::get_queue_length(num_thread);
            }

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
            std::int64_t get_average_thread_wait_time(
                std::size_t num_thread) const
            {
                return sched_->Scheduler::get_average_thread_wait_time(
                    num_thread);
            }

            std::int64_t get_average_task_wait_time(
                std::size_t num_thread) const
            {
                return sched_->Scheduler::get_average_task_wait_time(
                    num_thread);
            }
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
            std::int64_t get_num_pending_misses(std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_pending_misses(num, reset);
            }

            std::int64_t get_num_pending_accesses(std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_pending_accesses(num, reset);
            }

            std::int64_t get_num_stolen_from_pending(
                std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_stolen_from_pending(
                    num, reset);
            }

            std::int64_t get_num_stolen_to_pending(std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_stolen_to_pending(num, reset);
            }

            std::int64_t get_num_stolen_from_staged(std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_stolen_from_staged(
                    num, reset);
            }

            std::int64_t get_num_stolen_to_staged(std::size_t num, bool reset)
            {
                return sched_->Scheduler::get_num_stolen_to_staged(num, reset);
            }
#endif

            std::size_t get_os_thread_count() const
            {
                return threads_.size();
            }

        protected:
            std::vector<compat::thread> threads_;    // vector of OS-threads

            friend struct init_tss_helper<Scheduler>;

        private:
            // hold the used scheduler
            std::unique_ptr<Scheduler> sched_;
            std::size_t
                thread_offset_;    // is equal to the accumulated number of
                                   // threads in all pools preceding this pool
                // in the thread indexation. That means, that in order to know
                // the global index of a thread it owns, the pool has to compute:
                // global index = thread_offset_ + local index.
        };

        ///////////////////////////////////////////////////////////////////////////
        template <typename Scheduler>
        struct init_tss_helper
        {
            init_tss_helper(
                thread_pool_impl<Scheduler> &pool, std::size_t thread_num)
              : pool_(pool)
              , thread_num_(thread_num)
            {
                pool.notifier_.on_start_thread(thread_num);
                threads::get_thread_manager().init_tss(thread_num);
                pool.sched_->Scheduler::on_start_thread(
                    thread_num - pool.get_thread_offset());
            }
            ~init_tss_helper()
            {
                pool_.sched_->Scheduler::on_stop_thread(thread_num_);
                threads::get_thread_manager().deinit_tss();
                pool_.notifier_.on_stop_thread(thread_num_);
            }

            thread_pool_impl<Scheduler> &pool_;
            std::size_t thread_num_;
        };
    }    // namespace detail
}}    // namespace hpx::threads

#endif
