//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_REDUCE_MAX_AUG_12_2009_0552PM)
#define HPX_LCOS_REDUCE_MAX_AUG_12_2009_0552PM

#include <hpx/hpx_fwd.hpp>
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

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    class reduce_max : public lcos::base_lco
    {
    private:
        struct tag {};
        typedef hpx::util::spinlock_pool<tag> mutex_type;
    public:
        typedef components::managed_component<detail::reduce_max> wrapping_type;

        enum action {
            reduce_max_signal = 0,
            reduce_max_wait = 1
        };

        enum { value = components::component_base_lco };

        reduce_max()
          : in_(1), out_(1), value_(0)
          {}

        reduce_max(int in, int out=1, int value=0)
          : in_(in),
            out_(out),
            value_(value)
        {}

        ~reduce_max()
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

            std::cout << "Wait(): in_ = " << in_ << " out_ = " << out_ << std::endl;

            // Note: out_ is a lower bound, if more show up before the last
            // in_, then they all get signaled, if it shows up after the
            // last in_, it waits on the next round.

            // we need to get the self anew for each round as it might
            // get executed in a different thread from the previous one
            threads::thread_self& self = threads::get_self();

            // mark the thread as suspended before adding to the queue
            threads::thread_id_type id = self.get_thread_id();
            reinterpret_cast<threads::thread*>(id)->
                set_state(threads::marked_for_suspension);
            queue_.enqueue(id);

            --out_;

            if (in_ == 0 && out_ <= 0)
            {
                // We have all the signals we need, and enough waits too.
                threads::thread_id_type id = 0;
                while (queue_.dequeue(&id))
                    threads::set_thread_state(id, threads::pending);
            }

            {
                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
                self.yield(threads::suspended);
            }

            return value_;
        }

        void signal(int value)
        {
            mutex_type::scoped_lock l(this);

            std::cout << "signal(" << value << "): in_ = " << in_
                      << " out_ = " << out_ << std::endl;

            if (in_ > 0)
            {
                if (value > value_)
                    value_ = value;
                --in_;
            }

            if (in_ == 0 && out_ <= 0)
            {
                // We have all the signals we need, and enough waits too.
                threads::thread_id_type id = 0;
                while (queue_.dequeue(&id))
                    threads::set_thread_state(id, threads::pending);
            }
        }

        typedef hpx::actions::result_action0<
            reduce_max, int, reduce_max_wait, &reduce_max::wait
        > wait_action;

        typedef hpx::actions::action1<
            reduce_max, reduce_max_signal, int, &reduce_max::signal
        > signal_action;

    private:
        int in_, out_;
        int value_;
        boost::lockfree::fifo<threads::thread_id_type> queue_;
    };
}}}

namespace hpx { namespace lcos
{
    class reduce_max
    {
    public:
        typedef detail::reduce_max wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        reduce_max(int in, int out=1, int value=0)
          : impl_(new wrapping_type(new wrapped_type(in,out,value)))
          {}

        naming::id_type get_gid()
        {
            return impl_->get_gid();
        }

        int wait()
        {
            return (*impl_)->wait();
        }

        void signal(int value)
        {
            (*impl_)->signal(value);
        }

    private:
        boost::shared_ptr<wrapping_type> impl_;
    };

}}

#endif
