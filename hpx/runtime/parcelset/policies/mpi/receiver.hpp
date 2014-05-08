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
        receiver(MPI_Comm communicator, int tag)
          : communicator_(communicator)
          , tag_(tag)
          , next_(0)
          , recvd_chunks_(0)
        {}

        int sender_tag() const
        {
            header_.assert_valid();
            return header_.tag();
        }

        int tag() const
        {
            return tag_;
        }

        int rank() const
        {
            header_.assert_valid();
            return header_.rank();
        }

        void async_read(header const & h, connection_handler & parcelport)
        {
            header_ = h;
            buffer_ = get_buffer();
            buffer_->clear();
            next_ = 0;
            recvd_chunks_ = 0;

            // Store the time of the begin of the read operation
            performance_counters::parcels::data_point& data = buffer_->data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.serialization_time_ = 0;
            data.bytes_ = header_.size(); //-V101
            data.num_parcels_ = 0;

            if (static_cast<std::size_t>(header_.size()) > get_max_inbound_size(parcelport))
            {
                // report this problem ...
                HPX_THROW_EXCEPTION(boost::asio::error::operation_not_supported,
                    "mpi::receiver::receiver",
                    "The size of this message exceeds the maximum inbound data size");
                return;
            }

            header_.assert_valid();
            HPX_ASSERT(header_.rank() != util::mpi_environment::rank());

            buffer_->num_chunks_.first = header_.num_chunks_first();
            buffer_->num_chunks_.second = header_.num_chunks_second();

            next(&receiver::send_tag);
        }

        bool done(connection_handler & pp)
        {
            HPX_ASSERT(next_ != 0);
            if(((*this).*next_)())
            {
                // take measurement of overall receive time
                buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_->data_point_.time_;

                // decode the received parcels.
                decode_parcels(pp, *this, buffer_);
                return true;
            }
            return false;
        }

        ~receiver()
        {
        }

    private:

        bool send_tag()
        {
            MPI_Isend(
                &tag_
              , 1
              , MPI_INT
              , header_.rank()
              , header_.tag()
              , communicator_
              , &request_);
            return check_tag_sent();
        }

        bool check_tag_sent()
        {
            if(request_ready())
            {
                return start_send();
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
                buffer_->data_.resize(static_cast<std::size_t>(header_.size()));
                buffer_->chunks_.resize(num_zero_copy_chunks);
                return recv_transmission_chunks();
            }
            else
            {
                buffer_->data_.resize(std::size_t(header_.size()));
                return recv_data();
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
                header_.rank(),     // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            return check_transmission_chunks_recvd();
        }

        bool check_transmission_chunks_recvd()
        {
            if(request_ready())
            {
                return recv_data();
            }

            return next(&receiver::check_transmission_chunks_recvd);
        }

        bool recv_data()
        {
            MPI_Irecv(
                buffer_->data_.data(),    // data pointer
                static_cast<int>(buffer_->data_.size()), // number of elements
                MPI_CHAR,           // MPI Datatype
                header_.rank(),     // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            return check_data_recvd();
        }

        bool check_data_recvd()
        {
            if(request_ready())
            {
                return recv_chunks();
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
                return send_ack();
            }

            std::size_t chunk_size
                = buffer_->transmission_chunks_[recvd_chunks_].second;

            buffer_->chunks_[recvd_chunks_].resize(chunk_size);
            MPI_Irecv(
                buffer_->chunks_[recvd_chunks_].data(),  // data pointer
                static_cast<int>(chunk_size), // number of elements
                MPI_CHAR,           // MPI Datatype
                header_.rank(),     // Source
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
                return recv_chunks();
            }

            return next(&receiver::check_chunks_recvd);
        }

        bool send_ack()
        {
            MPI_Isend(
                &ack_,          // Data pointer
                1,              // Size
                MPI_CHAR,       // MPI Datatype
                header_.rank(), // Destination
                header_.tag(),  // Tag
                communicator_,  // Communicator
                &request_       // Request
            );
            return check_ack_sent();
        }

        bool check_ack_sent()
        {
            if(request_ready())
            {
                next(0);
                return true;
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

        header header_;
        MPI_Comm communicator_;
        int tag_;
        MPI_Request request_;
        int ack_;

        next_function_type next_;
        std::size_t recvd_chunks_;

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
