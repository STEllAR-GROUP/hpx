//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_HPP

#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/plugins/parcelport/mpi/mpi_environment.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct tag_provider
    {
        typedef lcos::local::spinlock mutex_type;

        struct tag
        {
            tag(tag_provider *provider)
              : provider_(provider)
              , tag_(provider_->acquire())
            {}

            operator int () const
            {
                return tag_;
            }

            ~tag()
            {
                provider_->release(tag_);
            }

            tag_provider *provider_;
            int tag_;
        };

        tag_provider()
          : next_tag_(1)
        {}

        tag operator()()
        {
            return tag(this);
        }

        int acquire()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            if(free_tags_.empty())
                return next_tag_++;

            int tag = free_tags_.front();
            free_tags_.pop_front();
            return tag;
        }

        void release(int tag)
        {
            if(tag == next_tag_) return;

            boost::lock_guard<mutex_type> l(mtx_);
            free_tags_.push_back(tag);
        }

        mutex_type mtx_;
        int next_tag_;
        std::deque<int> free_tags_;
    };

    struct sender
    {
        typedef hpx::lcos::local::spinlock mutex_type;
        sender(std::size_t max_connections)
          : max_connections_(max_connections)
          , num_connections_(0)
        {}

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

        template <typename Buffer, typename Background>
        void send(int dest, Buffer & buffer, Background background)
        {
            check_num_connections chk(this);
            tag_provider::tag tag(tag_provider_());

            header h(buffer, tag);
            h.assert_valid();

            MPI_Request request;
            MPI_Request *wait_request = NULL;
            {
                util::mpi_environment::scoped_lock l;
                MPI_Isend(
                    h.data()
                  , h.data_size_
                  , MPI_BYTE
                  , dest
                  , 0
                  , util::mpi_environment::communicator()
                  , &request
                );
                wait_request = &request;
            }

            std::vector<typename Buffer::transmission_chunk_type>& chunks =
                buffer.transmission_chunks_;
            if(!chunks.empty())
            {
                wait_done(wait_request, background);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Isend(
                        chunks.data()
                      , static_cast<int>(
                            chunks.size()
                          * sizeof(typename Buffer::transmission_chunk_type)
                        )
                      , MPI_BYTE
                      , dest
                      , tag
                      , util::mpi_environment::communicator()
                      , &request
                    );
                    wait_request = &request;
                }
            }

            if(!h.piggy_back())
            {
                wait_done(wait_request, background);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Isend(
                        buffer.data_.data()
                      , static_cast<int>(buffer.data_.size())
                      , MPI_BYTE
                      , dest
                      , tag
                      , util::mpi_environment::communicator()
                      , &request
                    );
                    wait_request = &request;
                }
            }

            for (serialization::serialization_chunk& c : buffer.chunks_)
            {
                if(c.type_ == serialization::chunk_type_pointer)
                {
                    wait_done(wait_request, background);
                    {
                        util::mpi_environment::scoped_lock l;
                        MPI_Isend(
                            const_cast<void *>(c.data_.cpos_)
                          , static_cast<int>(c.size_)
                          , MPI_BYTE
                          , dest
                          , tag
                          , util::mpi_environment::communicator()
                          , &request
                        );
                        wait_request = &request;
                    }
                }
            }

            wait_done(wait_request, background);
        }

    private:
        tag_provider tag_provider_;

        mutex_type connections_mtx_;
        lcos::local::detail::condition_variable connections_cond_;
        std::size_t const max_connections_;
        std::size_t num_connections_;

        template <typename Background>
        void wait_done(MPI_Request *& request, Background const & background)
        {
            if(request == NULL) return;

            std::size_t k = 0;

            while(true)
            {
                if(request_done(*request))
                {
                    break;
                }

                if (k < 4) //-V112
                {
                }
                else if(k < 32 || k & 1) //-V112
                {
                    if(threads::get_self_ptr())
                        this_thread::suspend(threads::pending,
                            "mpi::sender::wait_done");
                }
                else
                {
                    background();
                }
                ++k;
            }
            request = NULL;
        }

        bool request_done(MPI_Request & r)
        {
            util::mpi_environment::scoped_lock l;

            int completed = 0;
            MPI_Status status;
            int ret = 0;
            ret = MPI_Test(&r, &completed, &status);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if(completed)// && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                return true;
            }
            return false;
        }
    };

}}}}

#endif
