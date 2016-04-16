//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>
#include <hpx/plugins/parcelport/mpi/sender_connection.hpp>
#include <hpx/plugins/parcelport/mpi/tag_provider.hpp>

#include <iterator>
#include <list>
#include <memory>
#include <mutex>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct sender
    {
        typedef
            sender_connection
            connection_type;
        typedef boost::shared_ptr<connection_type> connection_ptr;
        typedef std::deque<connection_ptr> connection_list;

        typedef hpx::lcos::local::spinlock mutex_type;

        sender()
          : next_free_tag_(-1)
        {
        }

        void run()
        {
            get_next_free_tag();
        }

        connection_ptr create_connection(int dest,
            performance_counters::parcels::gatherer & parcels_sent)
        {
            return
                boost::make_shared<connection_type>(
                    this, dest, parcels_sent);
        }

        void add(connection_ptr const & ptr)
        {
            std::unique_lock<mutex_type> l(connections_mtx_);
            connections_.push_back(ptr);
        }

        int acquire_tag()
        {
            return tag_provider_.acquire();
        }

        void send_messages(
            connection_list connections
        )
        {
            // We try to handle all sends
            connection_list::iterator end = std::remove_if(
                connections.begin()
              , connections.end()
              , [](connection_ptr sender) -> bool
              {
                    if(sender->send())
                    {
                        error_code ec;
                        sender->postprocess_handler_(ec, sender->destination(), sender);
                        return true;
                    }
                    return false;
              }
            );

            // If some are still in progress, give them back
            if(connections.begin() != end)
            {
                std::unique_lock<mutex_type> l(connections_mtx_);
                connections_.insert(
                    connections_.end()
                  , std::make_move_iterator(connections.begin())
                  , std::make_move_iterator(end)
                );
            }
        }

        bool background_work()
        {
            connection_list connections;
            {
                std::unique_lock<mutex_type> l(connections_mtx_, std::try_to_lock);
                if(l && !connections_.empty())
                {
                    connections.push_back(connections_.front());
                    connections_.pop_front();
                }
            }
            bool has_work = false;
            if(!connections.empty())
            {
                send_messages(std::move(connections));
                has_work = true;
            }
            next_free_tag();
            return has_work;
        }

    private:
        tag_provider tag_provider_;

        void next_free_tag()
        {
            int next_free = -1;
            {
                std::unique_lock<mutex_type> l(next_free_tag_mtx_, std::try_to_lock);
                if(l)
                    next_free = next_free_tag_locked();
            }

            if(next_free != -1)
            {
                HPX_ASSERT(next_free > 1);
                tag_provider_.release(next_free);
            }
        }

        int next_free_tag_locked()
        {
            util::mpi_environment::scoped_try_lock l;

            if(l.locked)
            {
                MPI_Status status;
                int completed = 0;
                int ret = 0;
                ret = MPI_Test(&next_free_tag_request_, &completed, &status);
                HPX_ASSERT(ret == MPI_SUCCESS);
                if(completed)// && status->MPI_ERROR != MPI_ERR_PENDING)
                {
                    return get_next_free_tag();
                }
            }
            return -1;
        }

        int get_next_free_tag()
        {
            int next_free = next_free_tag_;
            MPI_Irecv(
                &next_free_tag_
              , 1
              , MPI_INT
              , MPI_ANY_SOURCE
              , 1
              , util::mpi_environment::communicator()
              , &next_free_tag_request_
            );
            return next_free;
        }

        mutex_type connections_mtx_;
        connection_list connections_;

        mutex_type next_free_tag_mtx_;
        MPI_Request next_free_tag_request_;
        int next_free_tag_;
    };


}}}}

#endif

#endif
