//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_HPP

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    class connection_handler;
    void add_sender(connection_handler & handler,
        boost::shared_ptr<sender> const& sender_connection);

    void close_sender_connection(connection_handler & handler, int tag, int rank);

    class sender
      : public parcelset::parcelport_connection<sender, std::vector<char> >
    {
    public:
        typedef
            HPX_STD_FUNCTION<void(boost::system::error_code const &, std::size_t)>
            handler_function_type;
        typedef
            HPX_STD_FUNCTION<
                void(
                    boost::system::error_code const &
                  , parcelset::locality const&
                  , boost::shared_ptr<sender>
                )
            >
            postprocess_function_type;

        typedef bool(sender::*next_function_type)();

        /// Construct a sending parcelport_connection with the given io_service.
        sender(MPI_Comm communicator,
            int tag,
            parcelset::locality const& locality_id,
            connection_handler & handler,
            performance_counters::parcels::gatherer& parcels_sent)
          : communicator_(communicator)
          , tag_(tag)
          , receiver_tag_(-1)
          , sent_chunks_(0)
          , next_(0)
          , parcelport_(handler)
          , there_(locality_id)
          , rank_(locality_id.get<locality>().rank())
          , parcels_sent_(parcels_sent)
        {}

        ~sender()
        {
            close_sender_connection(parcelport_, tag_, rank_);
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

        void verify(parcelset::locality const & parcel_locality_id) const
        {
            HPX_ASSERT(parcel_locality_id.get<locality>().rank() != util::mpi_environment::rank());
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            // Check for valid pre conditions
            HPX_ASSERT(next_ == 0);
            HPX_ASSERT(buffer_);
            HPX_ASSERT(!handler_);
            HPX_ASSERT(!postprocess_);

            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();

            handler_ = handler;
            postprocess_ = parcel_postprocess;

            // make header data structure
            header_ = header(
                rank_              // destination rank ...
              , tag_               // tag ...
              , *buffer_           // fill it with the buffer data ...
            );

            next(&sender::send_header);
            add_sender(parcelport_, shared_from_this());
        }

        bool done()
        {
            next_function_type f = 0;
            {
                hpx::lcos::local::spinlock::scoped_lock l(mtx_);
                f = next_;
            }
            if(f != 0)
            {
                if(((*this).*f)())
                {
                    verify_valid();
                    error_code ec;
                    handler_(ec, header_.size());
                    buffer_->data_point_.time_ = timer_.elapsed_nanoseconds()
                        - buffer_->data_point_.time_;
                    parcels_sent_.add_data(buffer_->data_point_);
                    // clear our state
                    buffer_.reset();
                    handler_.reset();
                    sent_chunks_ = 0;
                    postprocess_function_type pp;
                    std::swap(pp, postprocess_);
                    pp(ec, there_, shared_from_this());
                    return true;

                }
            }
            return false;
        }

    private:
        void verify_valid()
        {
            HPX_ASSERT(buffer_);
            HPX_ASSERT(handler_);
            HPX_ASSERT(postprocess_);
            header_.assert_valid();
            HPX_ASSERT(header_.rank() != util::mpi_environment::rank());
        }

        bool send_header()
        {
            verify_valid();
            MPI_Isend(
                header_.data(),     // Data pointer
                header_.data_size_, // Size
                header_.type(),     // MPI Datatype
                header_.rank(),     // Destination
                0,                  // Tag
                communicator_,      // Communicator
                &request_           // Request
            );
            return check_header_sent();
        }

        bool check_header_sent()
        {
            verify_valid();
            if(request_ready())
            {
                return recv_tag();
            }
            return next(&sender::check_header_sent);

        }

        bool recv_tag()
        {
            MPI_Irecv(
                &receiver_tag_, // data pointer
                1,              // number of elements
                MPI_INT,        // MPI Datatype
                header_.rank(), // Source
                tag_,           // Tag
                communicator_,  // Communicator
                &request_);     // Request
            return check_tag_recvd();
        }

        bool check_tag_recvd()
        {
            if(request_ready())
            {
                return send_transmission_chunks();
            }
            return next(&sender::check_tag_recvd);
        }

        bool send_transmission_chunks()
        {
            verify_valid();

            std::vector<parcel_buffer<buffer_type>::transmission_chunk_type>& chunks =
                buffer_->transmission_chunks_;
            if(chunks.empty())
            {
                return send_data();
            }

            MPI_Isend(
                chunks.data(), // Data pointer
                static_cast<int>(
                    chunks.size()
                        * sizeof(parcel_buffer<buffer_type>::transmission_chunk_type)
                    ), // Size
                MPI_CHAR,               // MPI Datatype
                header_.rank(),         // Destination
                receiver_tag_,          // Tag
                communicator_,          // Communicator
                &request_               // Request
                );
            return check_transmission_chunks_sent();
        }

        bool check_transmission_chunks_sent()
        {
            verify_valid();
            if(request_ready())
            {
                return send_data();
            }
            return next(&sender::check_transmission_chunks_sent);
        }

        bool send_data()
        {
            verify_valid();
            MPI_Isend(
                buffer_->data_.data(), // Data pointer
                static_cast<int>(buffer_->data_.size()), // Size
                MPI_CHAR,               // MPI Datatype
                header_.rank(),         // Destination
                receiver_tag_,          // Tag
                communicator_,          // Communicator
                &request_               // Request
                );
            return check_data_sent();
        }

        bool check_data_sent()
        {
            verify_valid();
            if(request_ready())
            {
                return send_chunks();
            }
            return next(&sender::check_data_sent);
        }

        bool send_chunks()
        {
            verify_valid();
            if(sent_chunks_ == buffer_->chunks_.size())
            {
                return recv_ack();
            }

            util::serialization_chunk & c = buffer_->chunks_[sent_chunks_];
            ++sent_chunks_;


            if(c.type_ == util::chunk_type_pointer)
            {
                MPI_Isend(
                    const_cast<void *>(c.data_.cpos_), // Data pointer
                    static_cast<int>(c.size_), // Size
                    MPI_CHAR,               // MPI Datatype
                    header_.rank(),         // Destination
                    receiver_tag_,          // Tag
                    communicator_,          // Communicator
                    &request_               // Request
                    );
                return check_chunks_sent();
            }
            else
            {
                return send_chunks();
            }
        }

        bool check_chunks_sent()
        {
            if(request_ready()) return send_chunks();
            return next(&sender::check_chunks_sent);
        }

        bool recv_ack()
        {
            MPI_Irecv(
                &ack_,              // data pointer
                1,                  // number of elements
                MPI_CHAR,           // MPI Datatype
                header_.rank(),     // Source
                tag_,               // Tag
                communicator_,      // Communicator
                &request_);         // Request
            return check_ack_recvd();
        }

        bool check_ack_recvd()
        {
            if(request_ready())
            {
                next(0);
                return true;
            }
            return next(&sender::check_ack_recvd);
        }

        bool request_ready()
        {
            int completed = 0;
            MPI_Status status;
            int ret = 0;
            ret = MPI_Test(&request_, &completed, &status);
            HPX_ASSERT(ret == MPI_SUCCESS);
            if(completed && status.MPI_ERROR != MPI_ERR_PENDING)
            {
                return true;
            }
            return false;
        }

        bool next(next_function_type f)
        {
            hpx::lcos::local::spinlock::scoped_lock l(mtx_);
            next_ = f;
            return false;
        }

        // This mutex protects the data members from possible races due to
        // concurrent access between async_write and done
        hpx::lcos::local::spinlock mtx_;
        MPI_Comm communicator_;

        header header_;
        int tag_;
        int receiver_tag_;
        char ack_;
        std::size_t sent_chunks_;
        next_function_type next_;

        MPI_Request request_;
        handler_function_type handler_;
        postprocess_function_type postprocess_;

        connection_handler & parcelport_;

        /// the other (receiving) end of this connection
        parcelset::locality there_;
        boost::int32_t rank_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };
}}}}

#endif
