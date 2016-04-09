//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM)
#define HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM

#include <hpx/config.hpp>
#include <hpx/state.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/assert.hpp>
#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#endif

#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include <hpx/config/warnings_prefix.hpp>

#include <boost/atomic.hpp>

#include <algorithm>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
    namespace detail
    {
        struct reset_on_exit
        {
            reset_on_exit(boost::atomic<boost::int32_t>& counter)
              : counter_(counter)
            {
                ++counter_;
                HPX_ASSERT(counter_ > 0);
            }
            ~reset_on_exit()
            {
                HPX_ASSERT(counter_ > 0);
                --counter_;
            }
            boost::atomic<boost::int32_t>& counter_;
        };
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// The scheduler_base defines the interface to be implemented by all
    /// scheduler policies
    struct scheduler_base
    {
    private:
        HPX_NON_COPYABLE(scheduler_base);

    public:
        scheduler_base(std::size_t num_threads,
                char const* description = "",
                scheduler_mode mode = nothing_special)
          : topology_(get_topology())
          , affinity_data_(num_threads)
          , mode_(mode)
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
          , wait_count_(0)
#endif
          , description_(description)
        {
            states_.resize(num_threads);
            for (std::size_t i = 0; i != num_threads; ++i)
                states_[i].store(state_initialized);
        }

        virtual ~scheduler_base()
        {
        }

        threads::mask_cref_type get_pu_mask(topology const& topology,
            std::size_t num_thread) const
        {
            return affinity_data_.get_pu_mask(topology, num_thread,
                this->numa_sensitive());
        }

        std::size_t get_pu_num(std::size_t num_thread) const
        {
            return affinity_data_.get_pu_num(num_thread);
        }

        void add_punit(std::size_t virt_core, std::size_t thread_num)
        {
            affinity_data_.add_punit(virt_core, thread_num, topology_);
        }

        std::size_t init(init_affinity_data const& data,
            topology const& topology)
        {
            return affinity_data_.init(data, topology);
        }

        void idle_callback(std::size_t /*num_thread*/)
        {
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
            // Put this thread to sleep for some time, additionally it gets
            // woken up on new work.
#if BOOST_VERSION < 105000
            boost::posix_time::millisec period(++wait_count_);

            boost::unique_lock<boost::mutex> l(mtx_);
            cond_.timed_wait(l, period);
#else
            boost::chrono::milliseconds period(++wait_count_);

            boost::unique_lock<boost::mutex> l(mtx_);
            cond_.wait_for(l, period);
#endif
#endif
        }

        bool background_callback(std::size_t num_thread)
        {
            bool result = false;
            if (hpx::parcelset::do_background_work(num_thread))
                result = true;

            if (0 == num_thread)
                hpx::agas::garbage_collect_non_blocking();
            return result;
        }

        /// This function gets called by the thread-manager whenever new work
        /// has been added, allowing the scheduler to reactivate one or more of
        /// possibly idling OS threads
        void do_some_work(std::size_t num_thread)
        {
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
            wait_count_.store(0, boost::memory_order_release);

            if (num_thread == std::size_t(-1))
                cond_.notify_all();
            else
                cond_.notify_one();
#endif
        }

        // allow to access/manipulate states
        boost::atomic<hpx::state>& get_state(std::size_t num_thread)
        {
            HPX_ASSERT(num_thread < states_.size());
            return states_[num_thread];
        }
        boost::atomic<hpx::state> const& get_state(std::size_t num_thread) const
        {
            HPX_ASSERT(num_thread < states_.size());
            return states_[num_thread];
        }

        void set_all_states(hpx::state s)
        {
            typedef boost::atomic<hpx::state> state_type;
            for (state_type& state : states_)
                state.store(s);
        }

        // return whether all states are at least at the given one
        bool has_reached_state(hpx::state s) const
        {
            typedef boost::atomic<hpx::state> state_type;
            for (state_type const& state : states_)
            {
                if (state.load() < s)
                    return false;
            }
            return true;
        }

        bool is_state(hpx::state s) const
        {
            typedef boost::atomic<hpx::state> state_type;
            for (state_type const& state : states_)
            {
                if (state.load() != s)
                    return false;
            }
            return true;
        }

        std::pair<hpx::state, hpx::state> get_minmax_state() const
        {
            std::pair<hpx::state, hpx::state> result(
                last_valid_runtime_state, first_valid_runtime_state);

            typedef boost::atomic<hpx::state> state_type;
            for (state_type const& state_iter : states_)
            {
                hpx::state s = state_iter.load();
                result.first = (std::min)(result.first, s);
                result.second = (std::max)(result.second, s);
            }

            return result;
        }

        // get/set scheduler mode
        scheduler_mode get_scheduler_mode() const
        {
            return mode_.load(boost::memory_order_acquire);
        }

        void set_scheduler_mode(scheduler_mode mode)
        {
            mode_.store(mode);
        }

        ///////////////////////////////////////////////////////////////////////
        virtual bool numa_sensitive() const { return false; }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        virtual boost::uint64_t get_creation_time(bool reset) = 0;
        virtual boost::uint64_t get_cleanup_time(bool reset) = 0;
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        virtual boost::int64_t get_num_pending_misses(std::size_t num_thread,
            bool reset) = 0;
        virtual boost::int64_t get_num_pending_accesses(std::size_t num_thread,
            bool reset) = 0;

        virtual boost::int64_t get_num_stolen_from_pending(std::size_t num_thread,
            bool reset) = 0;
        virtual boost::int64_t get_num_stolen_to_pending(std::size_t num_thread,
            bool reset) = 0;
        virtual boost::int64_t get_num_stolen_from_staged(std::size_t num_thread,
            bool reset) = 0;
        virtual boost::int64_t get_num_stolen_to_staged(std::size_t num_thread,
            bool reset) = 0;
#endif

        virtual boost::int64_t get_queue_length(
            std::size_t num_thread = std::size_t(-1)) const = 0;

        virtual boost::int64_t get_thread_count(
            thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1),
            bool reset = false) const = 0;

        virtual void abort_all_suspended_threads() = 0;

        virtual bool cleanup_terminated(bool delete_all = false) = 0;

        virtual void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread) = 0;

        virtual bool get_next_thread(std::size_t num_thread,
            boost::int64_t& idle_loop_count, threads::thread_data*& thrd) = 0;

        virtual void schedule_thread(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;
        virtual void schedule_thread_last(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;

        virtual bool destroy_thread(threads::thread_data* thrd,
            boost::int64_t& busy_count) = 0;

        virtual bool wait_or_add_new(std::size_t num_thread, bool running,
            boost::int64_t& idle_loop_count) = 0;

        virtual void on_start_thread(std::size_t num_thread) = 0;
        virtual void on_stop_thread(std::size_t num_thread) = 0;
        virtual void on_error(std::size_t num_thread,
            boost::exception_ptr const& e) = 0;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        virtual boost::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual boost::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
#endif

        virtual void start_periodic_maintenance(
            boost::atomic<hpx::state>& global_state)
        {}

        virtual void reset_thread_distribution() {}

    protected:
        topology const& topology_;
        detail::affinity_data affinity_data_;
        boost::atomic<scheduler_mode> mode_;

#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        // support for suspension on idle queues
        boost::mutex mtx_;
        boost::condition_variable cond_;
        boost::atomic<boost::uint32_t> wait_count_;
#endif

        boost::ptr_vector<boost::atomic<hpx::state> > states_;
        char const* description_;

#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
    public:
        coroutines::detail::tss_data_node* find_tss_data(void const* key)
        {
            if (!thread_data_)
                return 0;
            return thread_data_->find(key);
        }

        void add_new_tss_node(void const* key,
            boost::shared_ptr<coroutines::detail::tss_cleanup_function>
                const& func, void* tss_data)
        {
            if (!thread_data_)
            {
                thread_data_ =
                    boost::make_shared<coroutines::detail::tss_storage>();
            }
            thread_data_->insert(key, func, tss_data);
        }

        void erase_tss_node(void const* key, bool cleanup_existing)
        {
            if (thread_data_)
                thread_data_->erase(key, cleanup_existing);
        }

        void* get_tss_data(void const* key)
        {
            if (coroutines::detail::tss_data_node* const current_node =
                    find_tss_data(key))
            {
                return current_node->get_value();
            }
            return 0;
        }

        void set_tss_data(void const* key,
            boost::shared_ptr<coroutines::detail::tss_cleanup_function>
                const& func, void* tss_data, bool cleanup_existing)
        {
            if (coroutines::detail::tss_data_node* const current_node =
                    find_tss_data(key))
            {
                if (func || (tss_data != 0))
                    current_node->reinit(func, tss_data, cleanup_existing);
                else
                    erase_tss_node(key, cleanup_existing);
            }
            else if(func || (tss_data != 0))
            {
                add_new_tss_node(key, func, tss_data);
            }
        }

    protected:
        boost::shared_ptr<coroutines::detail::tss_storage> thread_data_;
#endif
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
