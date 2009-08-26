//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_PBREAK_AUG_25_2009_0829AM)
#define HPX_LCOS_PBREAK_AUG_25_2009_0829AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/full_empty_memory.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/variant.hpp>
#include <boost/static_assert.hpp>

#include <hpx/lcos/mutex.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>

#include <boost/assert.hpp>
#include <boost/lockfree/fifo.hpp>

#define LPBREAK_(lvl) LAPP_(lvl) << " [PBREAK] "

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    class pbreak : public lcos::base_lco
    {
    private:
        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;
    public:
        typedef components::managed_component<detail::pbreak> wrapping_type;

        enum action {
            pbreak_signal = 0,
            pbreak_wait = 1
        };

        enum { value = components::component_base_lco };

        pbreak()
          : seen_(0),
            total_(1)
          {}

        ~pbreak()
        {
            BOOST_ASSERT(queue_.empty());   // queue has to be empty
        }

        void set_event (void)
        {
            wait();
        }

        int wait(void)
        {
            mutex_type::scoped_lock l(this);

            LPBREAK_(info) << "Waiting: seen " << seen_ << ", total " << total_;

            // we need to get the self anew for each round as it might
            // get executed in a different thread from the previous one
            threads::thread_self& self = threads::get_self();

            // mark the thread as suspended before adding to the queue
            threads::thread_id_type id = self.get_thread_id();
            reinterpret_cast<threads::thread*>(id)->
                set_state(threads::marked_for_suspension);
            queue_.enqueue(id);

            if (seen_ == total_)
            {
                seen_ = total_ = 0;

                // We have all the signals we need
                threads::thread_id_type id = 0;
                while (queue_.dequeue(&id))
                    threads::set_thread_state(id, threads::pending);
            }

            {
                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            return 0;
        }

        void signal(int local_total)
        {
            mutex_type::scoped_lock l(this);

            LPBREAK_(info) << "Signaling(" << local_total << "): seen " << seen_ << ", total " << total_;

            if (local_total+1 > total_)
            {
                total_ = local_total;
            }

            ++seen_;
            if (seen_ == total_)
            {
                seen_ = 0;
                total_ = 1;

                // We have all the signals we need
                threads::thread_id_type id = 0;
                while (queue_.dequeue(&id))
                    threads::set_thread_state(id, threads::pending);
            }
        }

        typedef hpx::actions::result_action0<
            pbreak, int, pbreak_wait, &pbreak::wait
        > wait_action;

        typedef hpx::actions::action1<
            pbreak, pbreak_signal, int, &pbreak::signal
        > signal_action;

    private:
        int seen_;
        int total_;
        boost::lockfree::fifo<threads::thread_id_type> queue_;
    };
}}}

namespace hpx { namespace lcos
{
    class pbreak
    {
    public:
        typedef detail::pbreak wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        pbreak()
          : impl_(new wrapping_type(new wrapped_type()))
        {}

        naming::id_type get_gid()
        {
            return impl_->get_gid();
        }

        int wait()
        {
            return (*impl_)->wait();
        }

        void signal(int local_total)
        {
            (*impl_)->signal(local_total);
        }

    private:
        boost::shared_ptr<wrapping_type> impl_;
    };

}}

#endif
