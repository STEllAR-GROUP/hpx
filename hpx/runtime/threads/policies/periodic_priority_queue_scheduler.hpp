//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//  Copyright (c) 2011      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADMANAGER_SCHEDULING_PERIODIC_PRIORITY_QUEUE_HPP)
#define HPX_THREADMANAGER_SCHEDULING_PERIODIC_PRIORITY_QUEUE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace policies
{
    ///////////////////////////////////////////////////////////////////////////
    /// The periodic_priority_queue_scheduler maintains exactly one queue of work
    /// items (threads) per OS thread, where this OS thread pulls its next work
    /// from. Additionally it maintains separate queues: several for high
    /// priority threads and one for low priority threads.
    /// High priority threads are executed by the first N OS threads before any
    /// other work is executed. Low priority threads are executed by the last
    /// OS thread whenever no other work is available.
    template <typename Mutex
            , typename PendingQueuing
            , typename StagedQueuing
            , typename TerminatedQueuing
             >
    class periodic_priority_queue_scheduler
        : public local_priority_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
          >
    {
    public:
        typedef boost::mpl::true_ has_periodic_maintenance;

        typedef local_priority_queue_scheduler<
            Mutex, PendingQueuing, StagedQueuing, TerminatedQueuing
        > base_type;

        typedef typename base_type::thread_queue_type thread_queue_type;

        typedef typename base_type::init_parameter_type
            init_parameter_type;

        periodic_priority_queue_scheduler(init_parameter_type const& init)
          : base_type(init)
        {}

        bool periodic_maintenance(bool running)
        {
            // periodic maintenance redistributes work and is responsible that
            // every OS-Thread has enough work

            {
                // Calculate the average ...
                boost::int64_t average_task_count = 0;
                boost::int64_t average_work_count = 0;
                for(std::size_t i = 0; i < this->high_priority_queues_.size(); ++i)
                {
                    thread_queue_type* q = this->high_priority_queues_[i];
                    average_task_count += q->get_staged_queue_length();
                    average_work_count += q->get_pending_queue_length();
                }
                average_task_count =
                    average_task_count / this->high_priority_queues_.size();
                average_work_count =
                    average_work_count / this->high_priority_queues_.size();

                // Remove items from queues that have more than the average
                // FIXME: We should be able to avoid using a thread_queue as
                // a temporary.
                thread_queue_type tmp_queue;
                for(std::size_t i = 0; i < this->high_priority_queues_.size(); ++i)
                {
                    thread_queue_type* q = this->high_priority_queues_[i];
                    boost::int64_t task_items = q->get_staged_queue_length();
                    boost::int64_t work_items = q->get_pending_queue_length();
                    if(task_items > average_task_count)
                    {
                        boost::int64_t count = task_items - average_task_count;
                        tmp_queue.move_task_items_from(q, count);
                    }
                    if(work_items > average_work_count)
                    {
                        boost::int64_t count = work_items - average_work_count;
                        tmp_queue.move_work_items_from(q, count,
                                i + this->queues_.size());
                    }
                }

                // And re-add them to the queues which didn't have enough work ...
                for(std::size_t i = 0; i < this->high_priority_queues_.size(); ++i)
                {
                    thread_queue_type* q = this->high_priority_queues_[i];
                    boost::int64_t task_items = q->get_staged_queue_length();
                    boost::int64_t work_items = q->get_pending_queue_length();
                    if(task_items < average_task_count)
                    {
                        boost::int64_t count = average_task_count - task_items;
                        q->move_task_items_from(&tmp_queue, count);
                    }
                    if(work_items < average_work_count)
                    {
                        boost::int64_t count = average_work_count - work_items;
                        q->move_work_items_from(&tmp_queue, count,
                            i + this->queues_.size());
                    }
                }

                // Some items might remain in the tmp_queue ... readd them round robin
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_staged_queue_length())
                    {
                        this->high_priority_queues_[i]->
                            move_task_items_from(&tmp_queue, 1);
                        i = (i + 1) % this->high_priority_queues_.size();
                    }
                }
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_pending_queue_length())
                    {
                        this->high_priority_queues_[i]->
                            move_work_items_from(&tmp_queue, 1,
                                i + this->queues_.size());
                        i = (i + 1) % this->high_priority_queues_.size();
                    }
                }
            }

            {
                // Calculate the average ...
                boost::int64_t average_task_count = 0;
                boost::int64_t average_work_count = 0;
                for(std::size_t i = 0; i < this->queues_.size(); ++i)
                {
                    thread_queue_type* q = this->queues_[i];
                    average_task_count += q->get_staged_queue_length();
                    average_work_count += q->get_pending_queue_length();
                }
                average_task_count = average_task_count / this->queues_.size();
                average_work_count = average_work_count / this->queues_.size();

                // Remove items from queues that have more than the average
                thread_queue_type tmp_queue;
                for(std::size_t i = 0; i < this->queues_.size(); ++i)
                {
                    thread_queue_type* q = this->queues_[i];
                    boost::int64_t task_items = q->get_staged_queue_length();
                    boost::int64_t work_items = q->get_pending_queue_length();
                    if(task_items > average_task_count)
                    {
                        boost::int64_t count = task_items - average_task_count;
                        tmp_queue.move_task_items_from(q, count);
                    }
                    if(work_items > average_work_count)
                    {
                        boost::int64_t count = work_items - average_work_count;
                        tmp_queue.move_work_items_from(q, count,
                            i + this->queues_.size());
                    }
                }

                // And re-add them to the queues which didn't have enough work ...
                for(std::size_t i = 0; i < this->queues_.size(); ++i)
                {
                    thread_queue_type* q = this->queues_[i];
                    boost::int64_t task_items = q->get_staged_queue_length();
                    boost::int64_t work_items = q->get_pending_queue_length();
                    if(task_items < average_task_count)
                    {
                        boost::int64_t count = average_task_count - task_items;
                        q->move_task_items_from(&tmp_queue, count);
                    }
                    if(work_items < average_work_count)
                    {
                        boost::int64_t count = average_work_count - work_items;
                        q->move_work_items_from(&tmp_queue, count,
                            i + this->queues_.size());
                    }
                }
                // Some items might remain in the tmp_queue ... readd them round robin
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_staged_queue_length())
                    {
                        this->queues_[i]->move_task_items_from(&tmp_queue, 1);
                        i = (i + 1) % this->queues_.size();
                    }
                }
                {
                    std::size_t i = 0;
                    while(tmp_queue.get_pending_queue_length())
                    {
                        this->queues_[i]->move_work_items_from
                            (&tmp_queue, 1, i + this->queues_.size());
                        i = (i + 1) % this->queues_.size();
                    }
                }
            }

            return true;
        }
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
