//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_distributed/applier/trigger.hpp>
#include <hpx/async_distributed/base_lco_with_value.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <boost/intrusive/slist.hpp>

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace server {

    template <typename ValueType>
    struct object_semaphore
      : components::managed_component_base<object_semaphore<ValueType>>
    {
        using base_type = components::managed_component_base<object_semaphore>;
        using mutex_type = hpx::lcos::local::spinlock;

        // define data structures needed for intrusive slist container used for
        // the queues
        struct queue_thread_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>>
                hook_type;

            explicit queue_thread_entry(hpx::id_type const& id)
              : id_(id)
            {
            }

            hpx::id_type id_;
            hook_type slist_hook_;
        };

        using slist_option_type =
            boost::intrusive::member_hook<queue_thread_entry,
                typename queue_thread_entry::hook_type,
                &queue_thread_entry::slist_hook_>;

        using thread_queue_type = boost::intrusive::slist<queue_thread_entry,
            slist_option_type, boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>>;

        // queue holding the values to process
        struct queue_value_entry
        {
            typedef boost::intrusive::slist_member_hook<
                boost::intrusive::link_mode<boost::intrusive::normal_link>>
                hook_type;

            queue_value_entry(ValueType const& val, std::uint64_t count)
              : val_(val)
              , count_(count)
            {
            }

            ValueType val_;
            std::uint64_t count_;
            hook_type slist_hook_;
        };

        using value_slist_option_type =
            boost::intrusive::member_hook<queue_value_entry,
                typename queue_value_entry::hook_type,
                &queue_value_entry::slist_hook_>;

        using value_queue_type = boost::intrusive::slist<queue_value_entry,
            value_slist_option_type, boost::intrusive::cache_last<true>,
            boost::intrusive::constant_time_size<false>>;

    private:
        // assumes that this thread has acquired l
        void resume(std::unique_lock<mutex_type>& l)
        {
            HPX_ASSERT(l.owns_lock());

            // resume as many waiting LCOs as possible
            while (!thread_queue_.empty() && !value_queue_.empty())
            {
                ValueType value = value_queue_.front().val_;

                HPX_ASSERT(0 != value_queue_.front().count_);

                if (1 == value_queue_.front().count_)
                {
                    value_queue_.front().count_ = 0;
                    value_queue_.pop_front();
                }

                else
                    --value_queue_.front().count_;

                hpx::id_type id = thread_queue_.front().id_;
                thread_queue_.front().id_ = hpx::invalid_id;
                thread_queue_.pop_front();

                {
                    util::unlock_guard<std::unique_lock<mutex_type>> ul(l);

                    // set the LCO's result
                    applier::trigger(id, HPX_MOVE(value));
                }
            }
        }

    public:
        object_semaphore() = default;

        ~object_semaphore()
        {
            if (HPX_UNLIKELY(!thread_queue_.empty()))
                abort_pending(deadlock);
        }

        void signal(ValueType const& val, std::uint64_t count)
        {
            // push back the new value onto the queue
            std::unique_ptr<queue_value_entry> node(
                new queue_value_entry(val, count));

            std::unique_lock<mutex_type> l(mtx_);
            value_queue_.push_back(*node);

            node.release();

            resume(l);
        }

        void get(hpx::id_type const& lco)
        {
            // push the LCO's GID onto the queue
            std::unique_ptr<queue_thread_entry> node(
                new queue_thread_entry(lco));

            std::unique_lock<mutex_type> l(mtx_);

            thread_queue_.push_back(*node);

            node.release();

            resume(l);
        }

        void abort_pending(error ec)
        {
            std::lock_guard<mutex_type> l(mtx_);

            LLCO_(info).format("object_semaphore::abort_pending: thread_queue "
                               "is not empty, aborting threads");

            while (!thread_queue_.empty())
            {
                hpx::id_type id = thread_queue_.front().id_;
                thread_queue_.front().id_ = hpx::invalid_id;
                thread_queue_.pop_front();

                LLCO_(info).format(
                    "object_semaphore::abort_pending: pending thread {}", id);

                try
                {
                    HPX_THROW_EXCEPTION(ec, "object_semaphore::abort_pending",
                        "aborting pending thread");
                }
                catch (...)
                {
                    applier::trigger_error(id, std::current_exception());
                }
            }

            HPX_ASSERT(thread_queue_.empty());
        }

        void wait()
        {
            using action_type = typename lcos::template base_lco_with_value<
                ValueType>::get_value_action;

            std::lock_guard<mutex_type> l(mtx_);

            using const_iterator = typename thread_queue_type::const_iterator;
            const_iterator it = thread_queue_.begin(),
                           end = thread_queue_.end();

            for (; it != end; ++it)
            {
                hpx::id_type id = it->id_;

                LLCO_(info).format(
                    "object_semapohre::wait: waiting for {}", id);

                hpx::apply<action_type>(id);
            }
        }

        HPX_DEFINE_COMPONENT_ACTION(object_semaphore, signal, signal_action)
        HPX_DEFINE_COMPONENT_ACTION(object_semaphore, get, get_action)
        HPX_DEFINE_COMPONENT_ACTION(
            object_semaphore, abort_pending, abort_pending_action)
        HPX_DEFINE_COMPONENT_ACTION(object_semaphore, wait, wait_action)

    private:
        value_queue_type value_queue_;
        thread_queue_type thread_queue_;
        mutex_type mtx_;
    };
}}}    // namespace hpx::lcos::server
