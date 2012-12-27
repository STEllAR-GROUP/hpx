//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E)
#define HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E

#include <boost/intrusive/slist.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/trigger.hpp>
#include <hpx/lcos/base_lco.hpp>

namespace hpx { namespace lcos { namespace server
{

template <typename ValueType>
struct object_semaphore
  : components::managed_component_base<
        object_semaphore<ValueType>
    >
{
    typedef components::managed_component_base<object_semaphore> base_type;

    typedef hpx::lcos::local::spinlock mutex_type;

    // define data structures needed for intrusive slist container used for
    // the queues
    struct queue_thread_entry
    {
        typedef boost::intrusive::slist_member_hook<
            boost::intrusive::link_mode<boost::intrusive::normal_link>
        > hook_type;

        queue_thread_entry(naming::id_type const& id)
          : id_(id) {}

        naming::id_type id_;
        hook_type slist_hook_;
    };

    typedef boost::intrusive::member_hook<
        queue_thread_entry, typename queue_thread_entry::hook_type,
        &queue_thread_entry::slist_hook_
    > slist_option_type;

    typedef boost::intrusive::slist<
        queue_thread_entry, slist_option_type,
        boost::intrusive::cache_last<true>,
        boost::intrusive::constant_time_size<false>
    > thread_queue_type;

    // queue holding the values to process
    struct queue_value_entry
    {
        typedef boost::intrusive::slist_member_hook<
            boost::intrusive::link_mode<boost::intrusive::normal_link>
        > hook_type;

        queue_value_entry(ValueType const& val, boost::uint64_t count)
          : val_(val), count_(count) {}

        ValueType val_;
        boost::uint64_t count_;
        hook_type slist_hook_;
    };

    typedef boost::intrusive::member_hook<
        queue_value_entry, typename queue_value_entry::hook_type,
        &queue_value_entry::slist_hook_
    > value_slist_option_type;

    typedef boost::intrusive::slist<
        queue_value_entry, value_slist_option_type,
        boost::intrusive::cache_last<true>,
        boost::intrusive::constant_time_size<false>
    > value_queue_type;

  private:
    // assumes that this thread has acquired l
    void resume(mutex_type::scoped_lock& l)
    { // {{{
        // resume as many waiting LCOs as possible
        while (!thread_queue_.empty() && !value_queue_.empty())
        {
            ValueType value = value_queue_.front().val_;

            BOOST_ASSERT(0 != value_queue_.front().count_);

            if (1 == value_queue_.front().count_)
            {
                value_queue_.front().count_ = 0;
                value_queue_.pop_front();
            }

            else
                --value_queue_.front().count_;

            naming::id_type id = thread_queue_.front().id_;
            thread_queue_.front().id_ = naming::invalid_id;
            thread_queue_.pop_front();

            {
                util::unlock_the_lock<mutex_type::scoped_lock> ul(l);

                // set the LCO's result
                applier::trigger(id, boost::move(value));
            }
        }
    } // }}}

  public:
    object_semaphore() {}

    ~object_semaphore()
    { // {{{
        if (HPX_UNLIKELY(!thread_queue_.empty()))
            abort_pending(deadlock);
    } // }}}

    void signal(ValueType const& val, boost::uint64_t count)
    { // {{{
        // push back the new value onto the queue
        HPX_STD_UNIQUE_PTR<queue_value_entry> node
            (new queue_value_entry(val, count));

        mutex_type::scoped_lock l(mtx_);
        value_queue_.push_back(*node);

        node.release();

        resume(l);
    } // }}}

    void get(naming::id_type const& lco)
    { // {{{
        // push the LCO's GID onto the queue
        HPX_STD_UNIQUE_PTR<queue_thread_entry> node(new queue_thread_entry(lco));

        mutex_type::scoped_lock l(mtx_);

        thread_queue_.push_back(*node);

        node.release();

        resume(l);
    } // }}}

    void abort_pending(error ec)
    { // {{{
        mutex_type::scoped_lock l(mtx_);

        LLCO_(info)
            << "object_semaphore::abort_pending: thread_queue is not empty, "
               "aborting threads";

        while (!thread_queue_.empty())
        {
            naming::id_type id = thread_queue_.front().id_;
            thread_queue_.front().id_ = naming::invalid_id;
            thread_queue_.pop_front();

            LLCO_(info)
                << "object_semaphore::abort_pending: pending thread " << id;

            try
            {
                HPX_THROW_EXCEPTION(ec, "object_semaphore::abort_pending",
                    "aborting pending thread");
            }

            catch (hpx::exception const& /*e*/)
            {
                applier::trigger_error(id, boost::current_exception());
            }
        }

        BOOST_ASSERT(thread_queue_.empty());
    } // }}}

    void wait()
    { // {{{
        typedef typename
            lcos::template base_lco_with_value<ValueType>::get_value_action
        action_type;

        mutex_type::scoped_lock l(mtx_);

        typename thread_queue_type::const_iterator it = thread_queue_.begin()
                                                 , end = thread_queue_.end();

        for (; it != end; ++it)
        {
            naming::id_type id = it->id_;

            LLCO_(info) << "object_semapohre::wait: waiting for " << id;

            hpx::apply<action_type>(id);
        }
    } // }}}

    /*
    HPX_DEFINE_COMPONENT_ACTION_TPL(object_semaphore, signal, signal_action);
    */
    typedef
        hpx::actions::action2<
            object_semaphore<ValueType>
          , ValueType const&
          , boost::uint64_t
          , &object_semaphore<ValueType>::signal
        > signal_action;
    /*
    HPX_DEFINE_COMPONENT_ACTION_TPL(object_semaphore, get, get_action);
    */
    typedef
        hpx::actions::action1<
            object_semaphore<ValueType>
          , naming::id_type const& // lco
          , &object_semaphore<ValueType>::get
        > get_action;
    /*
    HPX_DEFINE_COMPONENT_ACTION_TPL(object_semaphore, abort_pending,
        abort_pending_action);
    */
    typedef
        hpx::actions::action1<
            object_semaphore<ValueType>
          , error
          , &object_semaphore<ValueType>::abort_pending
      > abort_pending_action;
    /*
    HPX_DEFINE_COMPONENT_ACTION_TPL(object_semaphore, wait, wait_action);
    */
    typedef
        hpx::actions::action0<
            object_semaphore<ValueType>
          , &object_semaphore<ValueType>::wait
        > wait_action;

  private:
    value_queue_type value_queue_;
    thread_queue_type thread_queue_;
    mutex_type mtx_;
};

}}}

#endif // HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E

