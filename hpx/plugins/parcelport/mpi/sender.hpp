//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_HPP

#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>
#include <hpx/plugins/parcelport/mpi/sender_connection.hpp>
#include <hpx/plugins/parcelport/mpi/tag_provider.hpp>

#include <list>
#include <iterator>

#include <boost/thread/locks.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    template <typename Buffer>
    struct sender
    {
        typedef
            sender_connection<typename hpx::util::decay<Buffer>::type>
            connection_type;
#if defined(HPX_INTEL_VERSION) && ((__GNUC__ == 4 && __GNUC_MINOR__ == 4) || HPX_INTEL_VERSION < 1400)
        typedef boost::shared_ptr<connection_type> connection_ptr;
#else
        typedef std::unique_ptr<connection_type> connection_ptr;
#endif
        typedef std::list<connection_ptr> connection_list;

        typedef hpx::lcos::local::spinlock mutex_type;
        sender(std::size_t max_connections)
          : max_connections_(max_connections)
          , num_connections_(0)
          , next_free_tag_(-1)
        {
        }

        void run()
        {
            get_next_free_tag();
        }

        struct check_num_connections
        {
            check_num_connections(sender *s)
             : this_(s)
             , decrement_(false)
            {
                if(!threads::get_self_ptr()) return;

                boost::unique_lock<mutex_type> l(this_->connections_mtx_);
                while(this_->num_connections_ >= this_->max_connections_)
                {
                    this_->connections_cond_.wait(l);
                }
                ++this_->num_connections_;
                decrement_ = true;
            }

            ~check_num_connections()
            {
                if(decrement_)
                {
                    boost::unique_lock<mutex_type> l(this_->connections_mtx_);
                    --this_->num_connections_;
                    this_->connections_cond_.notify_all(std::move(l));
                }
            }

            sender *this_;
            bool decrement_;
        };

        template <typename Handler>
        void send(
            int dest
          , parcel && p
          , Handler && handler
          , Buffer && buffer
          , performance_counters::parcels::gatherer & parcels_sent
        )
        {
            check_num_connections chk(this);

            connection_ptr sender(
                new connection_type(
                    tag_provider_.acquire()
                  , dest
                  , std::move(p)
                  , std::move(buffer)
                  , std::forward<Handler>(handler)
                  , parcels_sent
                )
            );

            if(!sender->send())
            {
                boost::unique_lock<mutex_type> l(connections_mtx_);
                connections_.push_back(std::move(sender));
            }
        }

        void send_messages(
            connection_list connections
        )
        {
            std::size_t k = 0;
            typename connection_list::iterator it = connections.begin();

            // We try to handle all receives within 1 secone
            while(it != connections.end())
            {
                connection_type & sender = **it;
                if(sender.send())
                {
                    it = connections.erase(it);
                }
                else
                {
                    if(k < 32 || k & 1) //-V112
                    {
                        if(threads::get_self_ptr())
                            hpx::this_thread::suspend(hpx::threads::pending,
                                "mpi::sender::wait_done");
                    }
                    ++k;
                    ++it;
                }
            }

            if(!connections.empty())
            {
                boost::unique_lock<mutex_type> l(connections_mtx_);
                if(connections_.empty())
                {
                    std::swap(connections, connections_);
                }
                else
                {
                    connections_.insert(
                        connections_.end()
#if defined(HPX_INTEL_VERSION) && ((__GNUC__ == 4 && __GNUC_MINOR__ == 4) || HPX_INTEL_VERSION < 1400)
                      , connections.begin()
                      , connections.end()
#else
                      , std::make_move_iterator(connections.begin())
                      , std::make_move_iterator(connections.end())
#endif
                    );
                }
            }
        }

        bool background_work(std::size_t num_thread)
        {
            connection_list connections;
            {
                boost::unique_lock<mutex_type> l(connections_mtx_);
                std::swap(connections, connections_);
            }
            bool has_work = false;
            if(!connections.empty())
            {
                if(hpx::is_starting())
                {
                    send_messages(std::move(connections));
                }
                else
                {
//                     error_code ec(lightweight);
                    hpx::applier::register_thread_nullary(
                        util::bind(
                            util::one_shot(&sender::send_messages),
                            this, std::move(connections)),
                        "mpi::sender::send_messages",
                        threads::pending, true, threads::thread_priority_boost,
                        num_thread, threads::thread_stacksize_default);
                }
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
                mutex_type::scoped_try_lock l(next_free_tag_mtx_);
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
        lcos::local::detail::condition_variable connections_cond_;
        std::size_t const max_connections_;
        std::size_t num_connections_;
        connection_list connections_;

        mutex_type next_free_tag_mtx_;
        MPI_Request next_free_tag_request_;
        int next_free_tag_;
    };

}}}}

#endif
