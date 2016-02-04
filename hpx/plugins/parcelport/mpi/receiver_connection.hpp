//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_RECEIVER_CONNECTION_HPP
#define HPX_PARCELSET_POLICIES_MPI_RECEIVER_CONNECTION_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_MPI)

#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    template <typename Parcelport>
    struct receiver_connection
    {
    private:
        enum connection_state
        {
            initialized
          , rcvd_transmission_chunks
          , rcvd_data
          , rcvd_chunks
          , sent_release_tag
        };

        typedef hpx::lcos::local::spinlock mutex_type;

        typedef std::vector<char>
            data_type;
        typedef parcel_buffer<data_type, data_type> buffer_type;

    public:
        receiver_connection(
            int src
          , header h
          , Parcelport & pp
        )
          : state_(initialized)
          , src_(src)
          , tag_(h.tag())
          , header_(h)
          , request_(MPI_REQUEST_NULL)
          , request_ptr_(0)
          , chunks_idx_(0)
          , pp_(pp)
        {
            header_.assert_valid();

            performance_counters::parcels::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.bytes_ = static_cast<std::size_t>(header_.numbytes());

            buffer_.data_.resize(static_cast<std::size_t>(header_.size()));
            buffer_.num_chunks_ = header_.num_chunks();
        }

        bool receive(std::size_t num_thread = -1)
        {
            switch (state_)
            {
                case initialized:
                    return receive_transmission_chunks(num_thread);
                case rcvd_transmission_chunks:
                    return receive_data(num_thread);
                case rcvd_data:
                    return receive_chunks(num_thread);
                case rcvd_chunks:
                    return send_release_tag(num_thread);
                case sent_release_tag:
                    return done();
                default:
                    HPX_ASSERT(false);
            }
            return false;
        }

        bool receive_transmission_chunks(std::size_t num_thread = -1)
        {
            // determine the size of the chunk buffer
            std::size_t num_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer_.num_chunks_.first));
            std::size_t num_non_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer_.num_chunks_.second));
            buffer_.transmission_chunks_.resize(
                num_zero_copy_chunks + num_non_zero_copy_chunks
            );
            if(num_zero_copy_chunks != 0)
            {
                buffer_.chunks_.resize(num_zero_copy_chunks);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Irecv(
                        buffer_.transmission_chunks_.data()
                      , static_cast<int>(
                            buffer_.transmission_chunks_.size()
                          * sizeof(buffer_type::transmission_chunk_type)
                        )
                      , MPI_BYTE
                      , src_
                      , tag_
                      , util::mpi_environment::communicator()
                      , &request_
                    );
                    request_ptr_ = &request_;
                }
            }

            state_ = rcvd_transmission_chunks;

            return receive_data(num_thread);
        }

        bool receive_data(std::size_t num_thread = -1)
        {
            if(!request_done()) return false;

            char *piggy_back = header_.piggy_back();
            if(piggy_back)
            {
                std::memcpy(&buffer_.data_[0], piggy_back, buffer_.data_.size());
            }
            else
            {
                util::mpi_environment::scoped_lock l;
                MPI_Irecv(
                    buffer_.data_.data()
                  , static_cast<int>(buffer_.data_.size())
                  , MPI_BYTE
                  , src_
                  , tag_
                  , util::mpi_environment::communicator()
                  , &request_
                );
                request_ptr_ = &request_;
            }

            state_ = rcvd_data;

            return receive_chunks(num_thread);
        }

        bool receive_chunks(std::size_t num_thread = -1)
        {
            while(chunks_idx_ < buffer_.chunks_.size())
            {
                if(!request_done()) return false;

                std::size_t idx = chunks_idx_++;
                std::size_t chunk_size = buffer_.transmission_chunks_[idx].second;

                data_type & c = buffer_.chunks_[idx];
                c.resize(chunk_size);
                {
                    util::mpi_environment::scoped_lock l;
                    MPI_Irecv(
                        c.data()
                      , static_cast<int>(c.size())
                      , MPI_BYTE
                      , src_
                      , tag_
                      , util::mpi_environment::communicator()
                      , &request_
                    );
                    request_ptr_ = &request_;
                }
            }

            state_ = rcvd_chunks;

            return send_release_tag(num_thread);
        }

        bool send_release_tag(std::size_t num_thread = -1)
        {
            if(!request_done()) return false;

            performance_counters::parcels::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds() - data.time_;

            {
                util::mpi_environment::scoped_lock l;
                MPI_Isend(
                    &tag_
                  , 1
                  , MPI_INT
                  , src_
                  , 1
                  , util::mpi_environment::communicator()
                  , &request_
                );
                request_ptr_ = &request_;
            }

            decode_parcels(pp_, std::move(buffer_), num_thread);

            state_ = sent_release_tag;

            return done();
        }

        bool done()
        {
            return request_done();
        }

        bool request_done()
        {
            if(request_ptr_ == 0) return true;

            util::mpi_environment::scoped_try_lock l;

            if(!l.locked) return false;

            int completed = 0;
            int ret = 0;
            ret = MPI_Test(request_ptr_, &completed, MPI_STATUS_IGNORE);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if(completed)// && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                request_ptr_ = 0;
                return true;
            }
            return false;
        }

        util::high_resolution_timer timer_;

        connection_state state_;

        int src_;
        int tag_;
        header header_;
        buffer_type buffer_;

        MPI_Request request_;
        MPI_Request *request_ptr_;
        std::size_t chunks_idx_;

        Parcelport & pp_;
    };
}}}}

#endif

#endif
