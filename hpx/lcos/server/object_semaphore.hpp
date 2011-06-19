//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E)
#define HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E

#include <boost/intrusive/slist.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/spinlock_pool.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/applier/trigger.hpp>
#include <hpx/lcos/dataflow_variable.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/get_result.hpp>

namespace hpx { namespace lcos { namespace server 
{

template <typename ValueType, typename RemoteType>
struct object_semaphore 
  : lcos::base_lco_with_value<
        boost::fusion::vector2<ValueType, boost::uint64_t>
      , boost::fusion::vector2<RemoteType, boost::uint64_t>
    >
  , components::managed_component_base<
        object_semaphore<ValueType, RemoteType>
    > 
{
    enum action
    {
        object_semaphore_add_lco = 4
    };

    typedef boost::fusion::vector2<ValueType, boost::uint64_t> value_type;
    typedef boost::fusion::vector2<RemoteType, boost::uint64_t> remote_type;
     
    typedef lcos::base_lco_with_value<value_type, remote_type> base_type_holder;

    typedef components::managed_component_base<object_semaphore> base_type;

    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;

    // define data structures needed for intrusive slist container used for
    // the queues
    struct queue_thread_entry
    {
        typedef boost::intrusive::slist_member_hook<
            boost::intrusive::link_mode<boost::intrusive::normal_link>
        > hook_type;

        queue_thread_entry(naming::id_type const& id)
          : id_(id), aborted_waiting_(false) {}

        naming::id_type id_;
        bool aborted_waiting_;
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

        queue_value_entry(value_type const& val)
          : val_(boost::fusion::at_c<0>(val))
          , count_(boost::fusion::at_c<1>(val)) {}

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
    void resume(typename mutex_type::scoped_lock& l)
    {       
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
                util::unlock_the_lock<typename mutex_type::scoped_lock> ul(l);

                // set the LCO's result 
                applier::trigger<ValueType>(id, value);  
            }
        }
    }

  public:
    object_semaphore() {}

    ~object_semaphore()
    {
        if (HPX_UNLIKELY(!thread_queue_.empty()))
        {
            try
            {
                HPX_THROW_EXCEPTION(deadlock, "~object_semaphore",
                    "semaphore is being destroyed with active LCOs");
            }

            catch (...)
            {
                set_error(boost::current_exception());
            }
        }
    }

    // disambiguate base classes
    using base_type::finalize;
    typedef typename base_type::wrapping_type wrapping_type;

    static components::component_type get_component_type()
    {
        return components::get_component_type<object_semaphore>();
    }
    static void set_component_type(components::component_type type) 
    {
        components::set_component_type<object_semaphore>(type);
    }

    void set_result(remote_type const& result)
    {
        // push back the new value onto the queue
        std::auto_ptr<queue_value_entry> node
            (new queue_value_entry
                (get_result<value_type, remote_type>::call(result)));

        typename mutex_type::scoped_lock l(this);
        value_queue_.push_back(*node);

        node.release();

        resume(l);
    }

    void set_error(boost::exception_ptr const& e)
    {
        typename mutex_type::scoped_lock l(this);

        LERR_(fatal)
            << "object_semaphore::set_error: thread_queue is not empty, "
               "aborting threads";

        while (!thread_queue_.empty())
        {
            naming::id_type id = thread_queue_.front().id_;
            thread_queue_.front().id_ = naming::invalid_id;
            thread_queue_.front().aborted_waiting_ = true;
            thread_queue_.pop_front();

            LERR_(fatal)
                << "object_semaphore::set_error: pending thread " << id; 

            // try to abort the thread, do not throw
            try
            {
                applier::trigger_error(id, e);
            }

            catch (...)
            {
                LERR_(fatal)
                    << "object_semaphore::set_error: "
                    << "could not abort thread " << id;
            }
        }

        BOOST_ASSERT(thread_queue_.empty());
    }

    // forwarder
    value_type get_value()
    {
        dataflow_variable<ValueType> data;
        naming::id_type lco = data.get_gid();

        add_lco(lco);

        return value_type(data.get(), 1);
    } 

    void add_lco(naming::id_type const& lco)
    {
        // push the LCO's GID onto the queue
        std::auto_ptr<queue_thread_entry> node(new queue_thread_entry(lco));
        
        typename mutex_type::scoped_lock l(this);

        thread_queue_.push_back(*node);

        node.release();

        resume(l);
    }

    typedef hpx::actions::direct_action1<
        object_semaphore<ValueType, RemoteType>
      , object_semaphore_add_lco
      , naming::id_type const& // lco
      , &object_semaphore<ValueType, RemoteType>::add_lco
    > add_lco_action;

  private:
    value_queue_type value_queue_;
    thread_queue_type thread_queue_;
};

}}}

#endif // HPX_1A262552_0D65_4C7D_887E_D11B02AAAC7E

