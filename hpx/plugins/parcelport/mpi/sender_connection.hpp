//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_CONNECTION_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_CONNECTION_HPP

#include <hpx/plugins/parcelport/mpi/header.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    template <typename Buffer>
    struct sender_connection
    {
    private:
        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;

        enum connection_state
        {
            initialized
          , sent_header
          , sent_transmission_chunks
          , sent_data
          , sent_chunks
        };

    public:
        sender_connection(
            int tag
          , int dst
          , parcel && p
          , Buffer && buffer
          , write_handler_type && handler
          , performance_counters::parcels::gatherer & parcels_sent
        )
          : state_(initialized)
          , tag_(tag)
          , dst_(dst)
          , parcel_(std::move(p))
          , buffer_(std::move(buffer))
          , handler_(std::move(handler))
          , header_(buffer_, tag_)
          , request_ptr_(0)
          , chunks_idx_(0)
          , parcels_sent_(parcels_sent)
        {
            header_.assert_valid();
        }

        bool send()
        {
            switch(state_)
            {
                case initialized:
                    return send_header();
                case sent_header:
                    return send_transmission_chunks();
                case sent_transmission_chunks:
                    return send_data();
                case sent_data:
                    return send_chunks();
                case sent_chunks:
                    return done();
                default:
                    HPX_ASSERT(false);
            }

            return false;
        }

        bool send_header()
        {
            {
                util::mpi_environment::scoped_lock l;
                HPX_ASSERT(state_ == initialized);
                HPX_ASSERT(request_ptr_ == 0);
                MPI_Isend(
                    header_.data()
                  , header_.data_size_
                  , MPI_BYTE
                  , dst_
                  , 0
                  , util::mpi_environment::communicator()
                  , &request_
                );
                request_ptr_ = &request_;
            }

            state_ = sent_header;
            return send_transmission_chunks();
        }

        bool send_transmission_chunks()
        {
            HPX_ASSERT(state_ == sent_header);
            HPX_ASSERT(request_ptr_ != 0);
            if(!request_done()) return false;

            HPX_ASSERT(request_ptr_ == 0);

            std::vector<typename Buffer::transmission_chunk_type>& chunks =
                buffer_.transmission_chunks_;
            if(!chunks.empty())
            {
                util::mpi_environment::scoped_lock l;
                MPI_Isend(
                    chunks.data()
                  , static_cast<int>(
                        chunks.size()
                      * sizeof(typename Buffer::transmission_chunk_type)
                    )
                  , MPI_BYTE
                  , dst_
                  , tag_
                  , util::mpi_environment::communicator()
                  , &request_
                );
                request_ptr_ = &request_;
            }

            state_ = sent_transmission_chunks;
            return send_data();
        }

        bool send_data()
        {
            HPX_ASSERT(state_ == sent_transmission_chunks);
            if(!request_done()) return false;

            if(!header_.piggy_back())
            {
                util::mpi_environment::scoped_lock l;
                MPI_Isend(
                    buffer_.data_.data()
                  , static_cast<int>(buffer_.data_.size())
                  , MPI_BYTE
                  , dst_
                  , tag_
                  , util::mpi_environment::communicator()
                  , &request_
                );
                request_ptr_ = &request_;
            }
            state_ = sent_data;

            return send_chunks();
        }

        bool send_chunks()
        {
            HPX_ASSERT(state_ == sent_data);

            while(chunks_idx_ < buffer_.chunks_.size())
            {
                serialization::serialization_chunk& c = buffer_.chunks_[chunks_idx_];
                if(c.type_ == serialization::chunk_type_pointer)
                {
                    if(!request_done()) return false;
                    else
                    {
                        util::mpi_environment::scoped_lock l;
                        MPI_Isend(
                            const_cast<void *>(c.data_.cpos_)
                          , static_cast<int>(c.size_)
                          , MPI_BYTE
                          , dst_
                          , tag_
                          , util::mpi_environment::communicator()
                          , &request_
                        );
                        request_ptr_ = &request_;
                    }
                 }

                chunks_idx_++;
            }

            if(!request_done()) return false;

            state_ = sent_chunks;

            return done();
        }

        bool done()
        {
            if(!request_done()) return false;

            error_code ec;
            handler_(ec, parcel_);
            buffer_.data_point_.time_ =
                util::high_resolution_clock::now() - buffer_.data_point_.time_;
            parcels_sent_.add_data(buffer_.data_point_);

            return true;
        }

        bool request_done()
        {
            if(request_ptr_ == 0) return true;

            util::mpi_environment::scoped_lock l;

            int completed = 0;
            MPI_Status status;
            int ret = 0;
            ret = MPI_Test(request_ptr_, &completed, &status);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if(completed)// && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                request_ptr_ = 0;
                return true;
            }
            return false;
        }

        connection_state state_;
        int tag_;
        int dst_;
        parcel parcel_;
        Buffer buffer_;
        write_handler_type handler_;

        header header_;

        MPI_Request request_;
        MPI_Request *request_ptr_;
        std::size_t chunks_idx_;
        char ack_;

        performance_counters::parcels::gatherer & parcels_sent_;
    };
}}}}

#endif

