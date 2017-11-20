//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM)
#define HPX_THREADMANAGER_SCHEDULING_SCHEDULER_BASE_JUL_14_2013_1132AM

#include <hpx/config.hpp>
#include <hpx/compat/condition_variable.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/state.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util_fwd.hpp>
#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
    namespace detail
    {
        struct reset_on_exit
        {
            reset_on_exit(std::atomic<std::int32_t>& counter)
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
            std::atomic<std::int32_t>& counter_;
        };
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    /// The scheduler_base defines the interface to be implemented by all
    /// scheduler policies
    struct scheduler_base
    {
    public:
        HPX_NON_COPYABLE(scheduler_base);

    public:
        scheduler_base(std::size_t num_threads,
                char const* description = "",
                scheduler_mode mode = nothing_special)
          : mode_(mode)
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
          , wait_count_(0)
#endif
          , states_(num_threads)
          , description_(description)
          , parent_pool_(nullptr)
        {
            for (std::size_t i = 0; i != num_threads; ++i)
                states_[i].store(state_initialized);
        }

        virtual ~scheduler_base()
        {
        }

        threads::detail::thread_pool_base *get_parent_pool()
        {
            HPX_ASSERT(parent_pool_ != nullptr);
            return parent_pool_;
        }

        void set_parent_pool(threads::detail::thread_pool_base *p)
        {
            HPX_ASSERT(parent_pool_ == nullptr);
            parent_pool_ = p;
        }

        inline std::size_t global_to_local_thread_index(std::size_t n)
        {
            return n - parent_pool_->get_thread_offset();
        }

        inline std::size_t local_to_global_thread_index(std::size_t n)
        {
            return n + parent_pool_->get_thread_offset();
        }

        char const* get_description() const { return description_; }

        void idle_callback(std::size_t /*num_thread*/)
        {
#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
            // Put this thread to sleep for some time, additionally it gets
            // woken up on new work.
            std::chrono::milliseconds period(++wait_count_);

            std::unique_lock<compat::mutex> l(mtx_);
            cond_.wait_for(l, period);
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
            wait_count_.store(0, std::memory_order_release);

            if (num_thread == std::size_t(-1))
                cond_.notify_all();
            else
                cond_.notify_one();
#endif
        }

        // allow to access/manipulate states
        std::atomic<hpx::state>& get_state(std::size_t num_thread)
        {
            HPX_ASSERT(num_thread < states_.size());
            return states_[num_thread];
        }
        std::atomic<hpx::state> const& get_state(std::size_t num_thread) const
        {
            HPX_ASSERT(num_thread < states_.size());
            return states_[num_thread];
        }

        void set_all_states(hpx::state s)
        {
            typedef std::atomic<hpx::state> state_type;
            for (state_type& state : states_)
                state.store(s);
        }

        // return whether all states are at least at the given one
        bool has_reached_state(hpx::state s) const
        {
            typedef std::atomic<hpx::state> state_type;
            for (state_type const& state : states_)
            {
                if (state.load() < s)
                    return false;
            }
            return true;
        }

        bool is_state(hpx::state s) const
        {
            typedef std::atomic<hpx::state> state_type;
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

            typedef std::atomic<hpx::state> state_type;
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
            return mode_.load(std::memory_order_acquire);
        }

        void set_scheduler_mode(scheduler_mode mode)
        {
            mode_.store(mode);
        }

        ///////////////////////////////////////////////////////////////////////
        virtual bool numa_sensitive() const { return false; }

        inline std::size_t domain_from_local_thread_index(std::size_t n)
        {
            auto &rp = resource::get_partitioner();
            auto const& topo = rp.get_topology();
            std::size_t global_id = local_to_global_thread_index(n);
            std::size_t pu_num = rp.get_pu_num(global_id);

            return topo.get_numa_node_number(pu_num);
        }

        template <typename Queue>
        std::size_t num_domains(const std::vector<Queue*> &queues)
        {
            auto &rp = resource::get_partitioner();
            auto const& topo = rp.get_topology();
            std::size_t num_queues = queues.size();

            std::set<std::size_t> domains;
            for (std::size_t local_id = 0; local_id != num_queues; ++local_id)
            {
                std::size_t global_id = local_to_global_thread_index(local_id);
                std::size_t pu_num = rp.get_pu_num(global_id);
                std::size_t dom = topo.get_numa_node_number(pu_num);
                domains.insert(dom);
            }
            return domains.size();
        }

        // either threads in same domain, or not in same domain
        // depending on the predicate
        std::vector<std::size_t> domain_threads(
            std::size_t local_id, const std::vector<std::size_t> &ts,
            std::function<bool(std::size_t, std::size_t)> pred)
        {
            std::vector<std::size_t> result;
            auto &rp = resource::get_partitioner();
            auto const& topo = rp.get_topology();
            std::size_t global_id = local_to_global_thread_index(local_id);
            std::size_t pu_num = rp.get_pu_num(global_id);
            std::size_t numa = topo.get_numa_node_number(pu_num);
            for (auto local_id : ts)
            {
                global_id = local_to_global_thread_index(local_id);
                pu_num = rp.get_pu_num(global_id);
                if (pred(numa, topo.get_numa_node_number(pu_num)))
                {
                    result.push_back(local_id);
                }
            }
            return result;
        }

#ifdef HPX_HAVE_THREAD_CREATION_AND_CLEANUP_RATES
        virtual std::uint64_t get_creation_time(bool reset) = 0;
        virtual std::uint64_t get_cleanup_time(bool reset) = 0;
#endif

#ifdef HPX_HAVE_THREAD_STEALING_COUNTS
        virtual std::int64_t get_num_pending_misses(std::size_t num_thread,
            bool reset) = 0;
        virtual std::int64_t get_num_pending_accesses(std::size_t num_thread,
            bool reset) = 0;

        virtual std::int64_t get_num_stolen_from_pending(std::size_t num_thread,
            bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_pending(std::size_t num_thread,
            bool reset) = 0;
        virtual std::int64_t get_num_stolen_from_staged(std::size_t num_thread,
            bool reset) = 0;
        virtual std::int64_t get_num_stolen_to_staged(std::size_t num_thread,
            bool reset) = 0;
#endif

        virtual std::int64_t get_queue_length(
            std::size_t num_thread = std::size_t(-1)) const = 0;

        virtual std::int64_t get_thread_count(
            thread_state_enum state = unknown,
            thread_priority priority = thread_priority_default,
            std::size_t num_thread = std::size_t(-1),
            bool reset = false) const = 0;

        // Enumerate all matching threads
        virtual bool enumerate_threads(
            util::function_nonser<bool(thread_id_type)> const& f,
            thread_state_enum state = unknown) const = 0;

        virtual void abort_all_suspended_threads() = 0;

        virtual bool cleanup_terminated(bool delete_all) = 0;
        virtual bool cleanup_terminated(std::size_t num_thread, bool delete_all) = 0;

        virtual void create_thread(thread_init_data& data, thread_id_type* id,
            thread_state_enum initial_state, bool run_now, error_code& ec,
            std::size_t num_thread) = 0;

        virtual bool get_next_thread(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count, threads::thread_data*& thrd) = 0;

        virtual void schedule_thread(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;
        virtual void schedule_thread_last(threads::thread_data* thrd,
            std::size_t num_thread,
            thread_priority priority = thread_priority_normal) = 0;

        virtual bool destroy_thread(threads::thread_data* thrd,
            std::int64_t& busy_count) = 0;

        virtual bool wait_or_add_new(std::size_t num_thread, bool running,
            std::int64_t& idle_loop_count) = 0;

        virtual void on_start_thread(std::size_t num_thread) = 0;
        virtual void on_stop_thread(std::size_t num_thread) = 0;
        virtual void on_error(std::size_t num_thread,
            std::exception_ptr const& e) = 0;

#ifdef HPX_HAVE_THREAD_QUEUE_WAITTIME
        virtual std::int64_t get_average_thread_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
        virtual std::int64_t get_average_task_wait_time(
            std::size_t num_thread = std::size_t(-1)) const = 0;
#endif

        virtual void start_periodic_maintenance(
            std::atomic<hpx::state>& global_state)
        {}

        virtual void reset_thread_distribution() {}

    protected:
        std::atomic<scheduler_mode> mode_;

#if defined(HPX_HAVE_THREAD_MANAGER_IDLE_BACKOFF)
        // support for suspension on idle queues
        compat::mutex mtx_;
        compat::condition_variable cond_;
        std::atomic<std::uint32_t> wait_count_;
#endif

        std::vector<std::atomic<hpx::state> > states_;
        char const* description_;

        // the pool that owns this scheduler
        threads::detail::thread_pool_base *parent_pool_;

#if defined(HPX_HAVE_SCHEDULER_LOCAL_STORAGE)
    public:
        coroutines::detail::tss_data_node* find_tss_data(void const* key)
        {
            if (!thread_data_)
                return nullptr;
            return thread_data_->find(key);
        }

        void add_new_tss_node(void const* key,
            std::shared_ptr<coroutines::detail::tss_cleanup_function>
                const& func, void* tss_data)
        {
            if (!thread_data_)
            {
                thread_data_ =
                    std::make_shared<coroutines::detail::tss_storage>();
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
            return nullptr;
        }

        void set_tss_data(void const* key,
            std::shared_ptr<coroutines::detail::tss_cleanup_function>
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
        std::shared_ptr<coroutines::detail::tss_storage> thread_data_;
#endif
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
