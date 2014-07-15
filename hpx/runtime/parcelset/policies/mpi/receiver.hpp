//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_MPI_RECEIVER_HPP

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <sstream>
#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class connection_handler;

    class receiver
      : public parcelport_connection<receiver, std::vector<char>, std::vector<char> >
    {
        typedef bool(receiver::*next_function_type)();
    public:
        receiver(MPI_Comm communicator, int tag, int sender_tag, int rank, connection_handler & parcelport)
          : communicator_(communicator)
          , tag_(tag)
          , sender_tag_(sender_tag)
          , rank_(rank)
          , next_(0)
          , recvd_chunks_(0)
          , pp_(parcelport)
          , closing_(false)
        {}

        int sender_tag() const
        {
            return sender_tag_;
        }

        int tag() const
        {
            return tag_;
        }

        int rank() const
        {
            return rank_;
        }

        void close()
        {
            closing_ = true;
        }

        bool closing()
        {
            if(closing_ && header_.empty())
            {
                closing_ = false;
                return true;
            }
            return false;
        }

        bool async_read(header const & h)
        {
            HPX_ASSERT(rank_ == h.rank());
            HPX_ASSERT(sender_tag_ == h.tag());

            header_.push_back(h);

            if(next_ == 0)
            {
                next(&receiver::prepare_send_tag);
                return true;
            }

            return false;
        }

        bool done()
        {
            HPX_ASSERT(next_ != 0);

            return ((*this).*next_)();
        }

        ~receiver()
        {
        }

    private:

        bool prepare_send_tag()
        {
            if(header_.empty())
            {
                next(0);
                return true;
            }

            current_header_ = header_.front();

            buffer_ = get_buffer();
            buffer_->clear();
            next_ = 0;
            recvd_chunks_ = 0;

            // Store the time of the begin of the read operation
            performance_counters::parcels::data_point& data = buffer_->data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.serialization_time_ = 0;
            data.bytes_ = current_header_.size(); //-V101
            data.num_parcels_ = 0;

            current_header_.assert_valid();
            HPX_ASSERT(rank_ != util::mpi_environment::rank());

            buffer_->num_chunks_.first = current_header_.num_chunks_first();
            buffer_->num_chunks_.second = current_header_.num_chunks_second();

            return next(&receiver::send_tag);
        }

        bool send_tag()
        {
            MPI_Isend(
                &tag_
              , 1
              , MPI_INT
              , rank_
              , sender_tag_
              , communicator_
              , &request_);
            return next(&receiver::check_tag_sent);
        }

        bool check_tag_sent()
        {
            if(request_ready())
            {
                return next(&receiver::start_send);
            }
            return next(&receiver::check_tag_sent);
        }

        bool start_send()
        {
            // determine the size of the chunk buffer
            std::size_t num_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer_->num_chunks_.first));
            std::size_t num_non_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer_->num_chunks_.second));

            if(num_zero_copy_chunks != 0)
            {
                buffer_->transmission_chunks_.resize(static_cast<std::size_t>(
                    num_zero_copy_chunks + num_non_zero_copy_chunks));
                buffer_->data_.resize(static_cast<std::size_t>(current_header_.size()));
                buffer_->chunks_.resize(num_zero_copy_chunks);
                return recv_transmission_chunks();
            }
            else
            {
                buffer_->data_.resize(std::size_t(current_header_.size()));
                return next(&receiver::recv_data);
            }
        }

        bool recv_transmission_chunks()
        {
            MPI_Irecv(
                buffer_->transmission_chunks_.data(),    // data pointer
                static_cast<int>(
                    buffer_->transmission_chunks_.size()
                        * sizeof(parcel_buffer_type::transmission_chunk_type)
                ),                  // number of elements
                MPI_CHAR,           // MPI Datatype
                rank_,              // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            return next(&receiver::check_transmission_chunks_recvd);
        }

        bool check_transmission_chunks_recvd()
        {
            if(request_ready())
            {
                return next(&receiver::recv_data);
            }

            return next(&receiver::check_transmission_chunks_recvd);
        }

        bool recv_data()
        {
            MPI_Irecv(
                buffer_->data_.data(),    // data pointer
                static_cast<int>(buffer_->data_.size()), // number of elements
                MPI_CHAR,           // MPI Datatype
                rank_,              // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            return next(&receiver::check_data_recvd);
        }

        bool check_data_recvd()
        {
            if(request_ready())
            {
                return next(&receiver::recv_chunks);
            }

            return next(&receiver::check_data_recvd);
        }

        bool recv_chunks()
        {
            // add appropriately sized chunk buffers for the zero-copy data
            std::size_t num_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<boost::uint32_t>(buffer_->num_chunks_.first));
            if(recvd_chunks_ == num_zero_copy_chunks)
            {
                return next(&receiver::send_ack);
            }

            std::size_t chunk_size
                = buffer_->transmission_chunks_[recvd_chunks_].second;

            buffer_->chunks_[recvd_chunks_].resize(chunk_size);
            MPI_Irecv(
                buffer_->chunks_[recvd_chunks_].data(),  // data pointer
                static_cast<int>(chunk_size), // number of elements
                MPI_CHAR,           // MPI Datatype
                rank_,              // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            ++recvd_chunks_;
            return next(&receiver::check_chunks_recvd);
        }

        bool check_chunks_recvd()
        {
            if(request_ready())
            {
                return next(&receiver::recv_chunks);
            }

            return next(&receiver::check_chunks_recvd);
        }

        bool send_ack()
        {
            MPI_Isend(
                &ack_,          // Data pointer
                1,              // Size
                MPI_CHAR,       // MPI Datatype
                rank_, // Destination
                sender_tag_,  // Tag
                communicator_,  // Communicator
                &request_       // Request
            );
            return next(&receiver::check_ack_sent);
        }

        bool check_ack_sent()
        {
            if(request_ready())
            {
                // take measurement of overall receive time
                buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_->data_point_.time_;

                // decode the received parcels.
                decode_parcels(pp_, *this, buffer_);
                header_.pop_front();
                current_header_ = header();

                return next(&receiver::prepare_send_tag);
            }

            return next(&receiver::check_ack_sent);
        }

        bool request_ready()
        {
            int completed = 0;
            MPI_Status status;
#ifdef HPX_DEBUG
            HPX_ASSERT(MPI_Test(&request_, &completed, &status) == MPI_SUCCESS);
#else
            MPI_Test(&request_, &completed, &status);
#endif
            if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                return true;
            }
            return false;
        }

        bool next(next_function_type f)
        {
            next_ = f;
            return false;
        }

        std::deque<header> header_;
        header current_header_;
        MPI_Comm communicator_;
        int tag_;
        int sender_tag_;
        int rank_;
        MPI_Request request_;
        int ack_;

        next_function_type next_;
        std::size_t recvd_chunks_;
        connection_handler & pp_;

        bool closing_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
    };

    // this makes sure we can store our connections in a set
    inline bool operator<(boost::shared_ptr<receiver> const& lhs,
        boost::shared_ptr<receiver> const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif
