
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP

#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>

#include <hpx/util/memory_chunk_pool.hpp>
#include <hpx/util/memory_chunk_pool_allocator.hpp>

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
        typedef util::detail::memory_chunk_pool_allocator<
                char, memory_pool_type
            > allocator_type;
        typedef
            std::vector<char, allocator_type>
            data_type;
        typedef parcel_buffer<data_type, data_type> buffer_type;

        struct handle_header
        {
            handle_header(header_list::iterator it, receiver &receiver,
                    boost::unique_lock<mutex_type> * l)
              : receiver_(receiver)
              , handles_(false)
              , l_(l)
            {
                // as handle_header tries to acquire a mutex as well, we need
                // to ignore the headers_mtx_ here
                util::ignore_while_checking<boost::unique_lock<mutex_type> > il(l_);
                boost::lock_guard<mutex_type> lk(receiver_.handles_header_mtx_);
                std::pair<int, int> p(it->first, it->second.tag());
                header_handle_ = receiver_.handles_header_.find(p);

                if(header_handle_ == receiver_.handles_header_.end())
                {
                    handles_ = true;
                    std::pair<handles_header_type::iterator, bool> ins_pair =
                        receiver_.handles_header_.insert(p);
                    HPX_ASSERT(ins_pair.second);
                    header_handle_ = ins_pair.first;
                }
            }

            handle_header(handle_header && other)
              : receiver_(other.receiver_)
              , handles_(other.handles_)
              , l_(other.l_)
              , header_handle_(other.header_handle_)
            {
                other.handles_ = false;
            }

            ~handle_header()
            {
                // as handle_header tries to acquire a mutex as well, we need
                // to ignore the headers_mtx_ here
                util::ignore_while_checking<boost::unique_lock<mutex_type> > il(l_);
                boost::lock_guard<mutex_type> lk(receiver_.handles_header_mtx_);
                if(handles_)
                {
                    HPX_ASSERT(header_handle_ != receiver_.handles_header_.end());

                    receiver_.handles_header_.erase(header_handle_);
                }
#if defined(HPX_DEBUG) && !defined(BOOST_MSVC)
                else
                {
                    HPX_ASSERT(header_handle_ != receiver_.handles_header_.end());
                }
#endif
            }

            receiver &receiver_;
            bool handles_;
            boost::unique_lock<mutex_type> * l_;
            handles_header_type::iterator header_handle_;
            HPX_MOVABLE_BUT_NOT_COPYABLE(handle_header);
        };

        receiver(parcelport & pp, memory_pool_type & chunk_pool
              , std::size_t max_connections)
          : pp_(pp)
          , chunk_pool_(chunk_pool)
          , max_connections_(max_connections)
          , num_connections_(0)
        {}

        void run()
        {
            new_header();
        }

        struct check_num_connections
        {
            check_num_connections(receiver *r)
             : this_(r)
             , decrement_(false)
            {
                if(this_->num_connections_ >= this_->max_connections_)
                    return;
                decrement_ = true;
                ++this_->num_connections_;
            }

            ~check_num_connections()
            {
                if(decrement_)
                {
                    --this_->num_connections_;
                }
            }

            receiver *this_;
            bool decrement_;
        };

        bool receive(bool background = true)
        {
            typedef header_list::iterator iterator;

            bool has_work = true;
            bool did_some_work = false;

            while(has_work)
            {
                check_num_connections chk(this);
                if(!chk.decrement_) break;
                boost::unique_lock<mutex_type> l(headers_mtx_);
                accept_locked();
                iterator it = headers_.begin();
                while(it != headers_.end())
                {
                    handle_header handle(it, *this, &l);
                    if(handle.handles_)
                    {

                        int source = it->first;
                        header h = it->second;
                        headers_.erase(it);

                        {
                            hpx::util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);

                            std::size_t num_thread(-1);
                            if(threads::get_self_ptr())
                                num_thread = hpx::get_worker_thread_num();

                            error_code ec(lightweight);
                            if(hpx::is_starting())
                            {
                                receive_message(source, h, std::move(handle));
                            }
                            else
                            {
                                hpx::applier::register_thread_nullary(
                                    util::bind(
                                        util::one_shot(&receiver::receive_message),
                                        this, source, h, std::move(handle)),
                                    "mpi::receiver::receive_message",
                                    threads::pending, true, threads::thread_priority_boost,
                                    num_thread, threads::thread_stacksize_default, ec);
                            }
                            h.reset();
                        }

                        if(background)
                            it = headers_.begin();
                        else
                            it = headers_.end();

                        did_some_work = true;
                    }
                    else
                    {
                        ++it;
                    }
                }
                if(it == headers_.end()) has_work = false;
            }
            return did_some_work;
        }

        void receive_message(int source, header h, handle_header)
        {
            util::high_resolution_timer timer;
            MPI_Request request;
            MPI_Request *wait_request = NULL;
            // Now do receives ...
            h.assert_valid();

            int tag = h.tag();

            allocator_type alloc(chunk_pool_);
            buffer_type buffer(alloc);

            performance_counters::parcels::data_point& data = buffer.data_point_;
            data.time_ = timer.elapsed_nanoseconds();
            data.bytes_ = static_cast<std::size_t>(h.numbytes());

            buffer.data_.resize(static_cast<std::size_t>(h.size()));
            buffer.num_chunks_ = h.num_chunks();

            // determine the size of the chunk buffer
            std::size_t num_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer.num_chunks_.first));
            std::size_t num_non_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer.num_chunks_.second));

            if(num_zero_copy_chunks != 0)
            {
                buffer.transmission_chunks_.resize(
                    num_zero_copy_chunks + num_non_zero_copy_chunks
                );
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Irecv(
                        buffer.transmission_chunks_.data()
                      , static_cast<int>(
                            buffer.transmission_chunks_.size()
                          * sizeof(buffer_type::transmission_chunk_type)
                        )
                      , MPI_BYTE
                      , source
                      , tag
                      , util::mpi_environment::communicator()
                      , &request
                    );
                    wait_request = &request;
                }
                buffer.chunks_.resize(num_zero_copy_chunks, data_type(alloc));
            }

            char *piggy_back = h.piggy_back();
            if(piggy_back)
            {
                std::memcpy(&buffer.data_[0], piggy_back, buffer.data_.size());
            }
            else
            {
                wait_done(wait_request);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Irecv(
                        buffer.data_.data()
                      , static_cast<int>(buffer.data_.size())
                      , MPI_BYTE
                      , source
                      , tag
                      , util::mpi_environment::communicator()
                      , &request
                    );
                    wait_request = &request;
                }
            }

            std::size_t chunk_idx = 0;
            for(data_type & c: buffer.chunks_)
            {
                wait_done(wait_request);
                std::size_t chunk_size = buffer.transmission_chunks_[chunk_idx++].second;

                c.resize(chunk_size);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Irecv(
                        c.data()
                      , static_cast<int>(c.size())
                      , MPI_BYTE
                      , source
                      , tag
                      , util::mpi_environment::communicator()
                      , &request
                    );
                    wait_request = &request;
                }
            }

            wait_done(wait_request);

            data.time_ = timer.elapsed_nanoseconds() - data.time_;

            decode_parcel(pp_, std::move(buffer));
        }

        void accept()
        {
            boost::unique_lock<mutex_type> l(headers_mtx_, boost::try_to_lock);
            if(l.owns_lock())
                accept_locked();
        }

        void accept_locked()
        {
            util::mpi_environment::scoped_try_lock l;

            if(l.locked)
            {
                MPI_Status status;
                if(request_done_locked(hdr_request_, &status))
                {
                    header h = new_header();
                    h.assert_valid();
                    headers_.push_back(std::make_pair(status.MPI_SOURCE, h));
                }
            }
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
        std::list<std::pair<int, header> > headers_;

        mutex_type handles_header_mtx_;
        handles_header_type handles_header_;

        mutex_type connections_mtx_;
        std::size_t const max_connections_;
        boost::atomic<std::size_t> num_connections_;

        void wait_done(MPI_Request *& request)
        {
            if(request == NULL) return;

            std::size_t k = 0;

            MPI_Status status;
            while(true)
            {
                if(request_done(*request, &status))
                {
                    break;
                }

                if (k < 4) //-V112
                {
                }
                else if(k < 32 || k & 1) //-V112
                {
                    if(threads::get_self_ptr())
                        hpx::this_thread::suspend(hpx::threads::pending,
                            "mpi::sender::wait_done");
                }
                else
                {
                    accept();
                }
                ++k;
            }
            request = NULL;
        }

        bool request_done(MPI_Request & r, MPI_Status *status)
        {
            util::mpi_environment::scoped_lock l;
            return request_done_locked(r, status);
        }

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
