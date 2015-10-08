
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP

#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/plugins/parcelport/mpi/receiver_connection.hpp>

#include <hpx/util/memory_chunk_pool.hpp>

#include <list>
#include <iterator>

#include <boost/thread/locks.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class parcelport;

    struct receiver
    {
        typedef hpx::lcos::local::spinlock mutex_type;
        typedef std::list<std::pair<int, header> > header_list;
        typedef std::set<std::pair<int, int> > handles_header_type;
        typedef util::memory_chunk_pool<mutex_type> memory_pool_type;
        typedef
            receiver_connection
            connection_type;
        typedef std::shared_ptr<connection_type> connection_ptr;
        typedef std::list<connection_ptr> connection_list;

        receiver(parcelport & pp, memory_pool_type & chunk_pool)
          : pp_(pp)
          , chunk_pool_(chunk_pool)
        {}

        void run()
        {
            util::mpi_environment::scoped_lock l;
            new_header();
        }

        bool background_work(std::size_t num_thread)
        {
            // We accept as many connections as we can ...
            connection_list connections;
            {
                boost::unique_lock<mutex_type> l(connections_mtx_);
                std::swap(connections, connections_);
            }

            connection_ptr rcv;
            do
            {
                rcv = accept();
                if(rcv && !rcv->receive())
                {
                    connections.push_back(rcv);
                }
            } while(rcv);
            rcv.reset();

            if(!connections.empty())
            {
                receive_messages(std::move(connections));
                return true;
            }

            return false;
        }

        void receive_messages(
            connection_list connections
        )
        {
            // We try to handle all receives
            connection_list::iterator end = std::remove_if(
                connections.begin()
              , connections.end()
              , [](connection_ptr & rcv) -> bool
              {
                    return rcv->receive();
              }
            );

            // If some are still in progress, give them back
//             if(end != connections.end())
            {
                boost::unique_lock<mutex_type> l(connections_mtx_);
                connections_.insert(
                    connections_.end()
                  , std::make_move_iterator(connections.begin())
                  , std::make_move_iterator(end)
                );
            }
        }

        connection_ptr accept()
        {
            boost::unique_lock<mutex_type> l(headers_mtx_, boost::try_to_lock);
            if(l)
                return accept_locked(l);
            return connection_ptr();
        }

        connection_ptr accept_locked(boost::unique_lock<mutex_type> & header_lock)
        {
            connection_ptr res;
            util::mpi_environment::scoped_try_lock l;

            if(l.locked)
            {
                MPI_Status status;
                if(request_done_locked(hdr_request_, &status))
                {
                    header h = new_header();
                    l.unlock();
                    header_lock.unlock();
                    res.reset(
                        new connection_type(
                            status.MPI_SOURCE
                          , h
                          , pp_
                          , chunk_pool_
                        )
                    );
                    return res;
                }
            }
            return res;
        }

        header new_header()
        {
            header h = rcv_header_;
            rcv_header_.reset();
            MPI_Irecv(
                rcv_header_.data()
              , rcv_header_.data_size_
              , MPI_BYTE
              , MPI_ANY_SOURCE
              , 0
              , util::mpi_environment::communicator()
              , &hdr_request_
            );
            return h;
        }

        parcelport & pp_;

        memory_pool_type & chunk_pool_;

        mutex_type headers_mtx_;
        MPI_Request hdr_request_;
        header rcv_header_;

        mutex_type handles_header_mtx_;
        handles_header_type handles_header_;

        mutex_type connections_mtx_;
        connection_list connections_;

        bool request_done_locked(MPI_Request & r, MPI_Status *status)
        {
            int completed = 0;
            int ret = 0;
            ret = MPI_Test(&r, &completed, status);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if(completed)// && status->MPI_ERROR != MPI_ERR_PENDING)
            {
                return true;
            }
            return false;
        }
    };

}}}}

#endif
