//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM)
#define HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM

#include <boost/version.hpp>
#include <boost/intrusive/slist.hpp>

#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/traits/get_remote_result.hpp>

#include <memory>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    /// A queue can be used to 'collect' (queue) a number of incoming values
    /// for consumption of an internal thread, which will invoke a given action
    /// for each of the values.
    template <typename ValueType, typename RemoteType = ValueType>
    class queue;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ValueType, typename RemoteType>
    class queue
      : public lcos::base_lco_with_value<ValueType, RemoteType>
      , public components::managed_component_base<queue<ValueType, RemoteType> >
    {
    public:
        typedef lcos::base_lco_with_value<ValueType, RemoteType> base_type_holder;

    private:
        typedef lcos::local::spinlock mutex_type;
        typedef components::managed_component_base<queue> base_type;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_thread_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_thread_entry(threads::thread_id_type id)
              : id_(id)
            {}

            threads::thread_id_type id_;
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

        struct reset_queue_entry
        {
            reset_queue_entry(queue_thread_entry& e, thread_queue_type& q)
              : e_(e), q_(q), last_(q.last())
            {}

            ~reset_queue_entry()
            {
                if (e_.id_)
                    q_.erase(last_);     // remove entry from queue
            }

            queue_thread_entry& e_;
            thread_queue_type& q_;
            typename thread_queue_type::const_iterator last_;
        };

        // queue holding the values to process
        struct queue_value_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_value_entry(ValueType const& val)
              : val_(val)
            {}

            ValueType val_;
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

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        queue()
        {}

        ~queue()
        {
            if (!thread_queue_.empty()) {
                LERR_(fatal) << "~queue: thread_queue is not empty, aborting threads";

                mutex_type::scoped_lock l(mtx_);
                while (!thread_queue_.empty()) {
                    threads::thread_id_type id = thread_queue_.front().id_;
                    thread_queue_.front().id_ = 0;
                    thread_queue_.pop_front();

                    // we know that the id is actually the pointer to the thread
                    threads::thread_data_base* thrd =
                        static_cast<threads::thread_data_base*>(id);
                    LERR_(fatal) << "~queue: pending thread: "
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();

                    // forcefully abort thread, do not throw
                    error_code ec(lightweight);
                    threads::set_thread_state(id, threads::pending,
                        threads::wait_abort, threads::thread_priority_default, ec);
                    if (ec) {
                        LERR_(fatal) << "~queue: could not abort thread"
                            << get_thread_state_name(thrd->get_state())
                            << "(" << id << "): " << thrd->get_description();
                    }
                }
            }
            BOOST_ASSERT(value_queue_.empty());
        }

        // disambiguate base classes
        using base_type::finalize;
        typedef typename base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<queue>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<queue>(type);
        }

        // standard LCO action implementations

        /// Add a value to the queue.
        void set_value (BOOST_RV_REF(RemoteType) result)
        {
            // push back the new value onto the queue
            HPX_STD_UNIQUE_PTR<queue_value_entry> node(
                new queue_value_entry(
                    traits::get_remote_result<ValueType, RemoteType>::call(result)));

            mutex_type::scoped_lock l(mtx_);
            value_queue_.push_back(*node);

            node.release();

            // resume the first thread waiting to pick up that value
            if (!thread_queue_.empty()) {
                threads::thread_id_type id = thread_queue_.front().id_;
                thread_queue_.front().id_ = 0;
                thread_queue_.pop_front();

                threads::set_thread_state(id, threads::pending);
            }
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& /*e*/)
        {
            mutex_type::scoped_lock l(mtx_);

            while (!thread_queue_.empty()) {
                threads::thread_id_type id = thread_queue_.front().id_;
                thread_queue_.front().id_ = 0;
                thread_queue_.pop_front();

                threads::set_thread_state(id, threads::pending, threads::wait_abort);
            }
        }

        // Retrieve the next value from the queue (pop value from front of
        // queue). This method blocks if the value queue is empty. Waiting
        // threads are resumed automatically as soon as new values are placed
        // into the value queue.
        ValueType get_value()
        {
            threads::thread_self& self = threads::get_self();

            mutex_type::scoped_lock l(mtx_);
            if (value_queue_.empty()) {
                // suspend this thread until a new value is placed into the
                // value queue
                queue_thread_entry e(self.get_thread_id());
                thread_queue_.push_back(e);

                reset_queue_entry r(e, thread_queue_);
                {
                    util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                    this_thread::suspend(threads::suspended,
                        "queue::get_value");
                }
            }

            // get the first value from the value queue and return it to the
            // caller
            ValueType value = value_queue_.front().val_;
            value_queue_.pop_front();

            return value;
        }

    private:
        mutex_type mtx_;
        value_queue_type value_queue_;
        thread_queue_type thread_queue_;
    };
}}}

#endif

