//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM)
#define HPX_LCOS_SERVER_QUEUE_FEB_09_2011_1204PM

#include <boost/intrusive/slist.hpp>

#include <hpx/exception.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
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

        // queue holding the values to process
        struct queue_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>
            > hook_type;

            queue_entry(ValueType const& val)
              : val_(val)
            {}

            ValueType val_;
            hook_type slist_hook_;
        };

        typedef boost::intrusive::member_hook<
            queue_entry, typename queue_entry::hook_type,
            &queue_entry::slist_hook_
        > slist_option_type;

        typedef boost::intrusive::slist<
            queue_entry, slist_option_type,
            boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>
        > queue_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        queue()
        {}

        ~queue()
        {
            HPX_ASSERT(queue_.empty());
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
        void set_value (RemoteType && result)
        {
            // push back the new value onto the queue
            std::unique_ptr<queue_entry> node(
                new queue_entry(
                    traits::get_remote_result<ValueType, RemoteType>::call(result)));

            mutex_type::scoped_lock l(mtx_);
            queue_.push_back(*node);

            node.release();

            // resume the first thread waiting to pick up that value
            cond_.notify_one(l);
        }

        /// The \a function set_exception is called whenever a
        /// \a set_exception_action is applied on an instance of a LCO.
        ///
        /// \param e      [in] The exception encapsulating the error to report
        ///               to this LCO instance.
        void set_exception(boost::exception_ptr const& /*e*/)
        {
            mutex_type::scoped_lock l(mtx_);
            cond_.abort_all(l);
        }

        // Retrieve the next value from the queue (pop value from front of
        // queue). This method blocks if the value queue is empty. Waiting
        // threads are resumed automatically as soon as new values are placed
        // into the value queue.
        ValueType get_value()
        {
            mutex_type::scoped_lock l(mtx_);
            if (queue_.empty()) {
                cond_.wait(l, "queue::get_value");
            }

            // get the first value from the value queue and return it to the
            // caller
            ValueType value = queue_.front().val_;
            queue_.pop_front();

            return value;
        }

    private:
        mutex_type mtx_;
        queue_type queue_;
        local::detail::condition_variable cond_;
    };
}}}

#endif

