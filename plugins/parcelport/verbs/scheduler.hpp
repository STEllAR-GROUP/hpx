#ifndef __PARCELPORT_VERBS_CUSTOM_SCHEDULER_H__
#define __PARCELPORT_VERBS_CUSTOM_SCHEDULER_H__
//
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/detail/create_thread.hpp>
#include <hpx/runtime/threads/detail/create_work.hpp>
//
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
//
#include <hpx/runtime/threads/policies/static_queue_scheduler.hpp>
//
#include <plugins/parcelport/verbs/rdma/rdma_logging.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
//
using namespace hpx;
using namespace hpx::threads::policies;
//
//----------------------------------------------------------------------------
namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs
{
    class custom_scheduler {
    private:
        typedef hpx::lcos::local::spinlock                               mutex_type;
        typedef hpx::parcelset::policies::verbs::scoped_lock<mutex_type> scoped_lock;
        typedef hpx::parcelset::policies::verbs::unique_lock<mutex_type> unique_lock;

    public:
        //    typedef static_queue_scheduler<
        //                boost::mutex, lockfree_fifo, lockfree_fifo, lockfree_lifo>
        //                scheduling_policy_type;
        typedef static_queue_scheduler<> scheduling_policy_type;

        // mutex protecting the members
        mutable boost::mutex mtx_;

        // some params we need to initialize the scheduler
        scheduling_policy_type::init_parameter init_param_;

        // The scheduler
        scheduling_policy_type scheduler_;

        // Callback notifier
        callback_notifier notifier_;

        // The thread pool we run the scheduler on
        hpx::threads::detail::thread_pool<scheduling_policy_type> pool_;

        // avoid warnings about usage of this in member initializer list
        custom_scheduler* This() { return this; }

        //----------------------------------------------------------------------------
        custom_scheduler() :
            init_param_(),
            scheduler_(init_param_, false), // false = not deferred initialization
            notifier_(get_runtime().get_notification_policy("custom-pool")),
            pool_(scheduler_, notifier_, "custom-pool")
        {
            LOG_DEBUG_MSG("Creating custom scheduler");
        }

        //----------------------------------------------------------------------------
        // a simple function that runs our hpx thread/function
        static inline threads::thread_result_type thread_function_nullary(
            util::unique_function_nonser<void()> func)
        {
            // execute the actual thread function
            func();

            // Verify that there are no more registered locks for this
            // OS-thread. This will throw if there are still any locks
            // held.
            util::force_error_on_lock();

            return threads::thread_result_type(threads::terminated, nullptr);
        }

        //----------------------------------------------------------------------------
        void init() {
            std::unique_lock<boost::mutex> lk(mtx_);
            init_affinity_data affinity_data;
            pool_.init(1, affinity_data);
            pool_.run(lk, 1);
        }

        //----------------------------------------------------------------------------
        void stop()
        {
            std::unique_lock<boost::mutex> lk(mtx_);
            pool_.stop(lk, true);
        }

        //----------------------------------------------------------------------------
        hpx::threads::thread_id_type register_thread_nullary(
            util::unique_function_nonser<void()> && func,
            util::thread_description const& desc,
            threads::thread_state_enum initial_state, bool run_now,
            threads::thread_priority priority, std::size_t os_thread,
            threads::thread_stacksize stacksize, error_code& ec)
        {
            util::thread_description d =
                desc ? desc : util::thread_description(func, "mystery thread");

            hpx::threads::thread_init_data data(
                util::bind(util::one_shot(&thread_function_nullary), std::move(func)),
                d, 0, priority, os_thread, threads::get_stack_size(stacksize));

            hpx::threads::thread_id_type id = threads::invalid_thread_id;

            hpx::threads::detail::create_thread(&scheduler_, data, id, initial_state, run_now, ec);

            std::cout << "created a thread and got id " << id << std::endl;
            return id;
        }

    };

}}}}

#endif
