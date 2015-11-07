//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_MPI_SENDER_CONNECTION_HPP
#define HPX_PARCELSET_POLICIES_MPI_SENDER_CONNECTION_HPP

#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/plugins/parcelport/mpi/header.hpp>
#include <hpx/plugins/parcelport/mpi/locality.hpp>

#include <boost/shared_ptr.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace mpi
{
    struct sender;
    struct sender_connection;

    int acquire_tag(sender *);
    void add_connection(sender *, boost::shared_ptr<sender_connection> const&);

    struct sender_connection
      : parcelset::parcelport_connection<
            sender_connection
          , std::vector<char>
        >
    {
    private:
        typedef sender sender_type;

        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;

        typedef std::vector<char> data_type;

        enum connection_state
        {
            initialized
          , sent_header
          , sent_transmission_chunks
          , sent_data
          , sent_chunks
        };

        typedef
            parcelset::parcelport_connection<sender_connection, data_type>
            base_type;

    public:
        sender_connection(
            sender_type * s
          , int dst
          , performance_counters::parcels::gatherer & parcels_sent
        )
          : state_(initialized)
          , sender_(s)
          , dst_(dst)
          , request_(MPI_REQUEST_NULL)
          , request_ptr_(0)
          , chunks_idx_(0)
          , ack_(0)
          , parcels_sent_(parcels_sent)
          , there_(
                parcelset::locality(
                    locality(
                        dst_
                    )
                )
            )
        {
        }

        parcelset::locality const& destination() const
        {
            return there_;
        }

        void verify(parcelset::locality const & parcel_locality_id) const
        {
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler && handler, ParcelPostprocess && parcel_postprocess)
        {
            HPX_ASSERT(!buffer_.data_.empty());
            request_ptr_ = 0;
            chunks_idx_ = 0;
            tag_ = acquire_tag(sender_);
            header_ = header(buffer_, tag_);
            header_.assert_valid();

            state_ = initialized;

            handler_ = std::forward<Handler>(handler);

            if(!send())
            {
                postprocess_handler_
                    = std::forward<ParcelPostprocess>(parcel_postprocess);
                add_connection(sender_, shared_from_this());
            }
            else
            {
                error_code ec;
                parcel_postprocess(ec, there_, shared_from_this());
            }
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

            std::vector<typename parcel_buffer_type::transmission_chunk_type>& chunks =
                buffer_.transmission_chunks_;
            if(!chunks.empty())
            {
                util::mpi_environment::scoped_lock l;
                MPI_Isend(
                    chunks.data()
                  , static_cast<int>(
                        chunks.size()
                      * sizeof(parcel_buffer_type::transmission_chunk_type)
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
            handler_(ec);
            buffer_.data_point_.time_ =
                util::high_resolution_clock::now() - buffer_.data_point_.time_;
            parcels_sent_.add_data(buffer_.data_point_);
            buffer_.clear();

            return true;
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

        connection_state state_;
        sender_type * sender_;
        int tag_;
        int dst_;
        util::unique_function_nonser<
            void(
                error_code const&
            )
        > handler_;
        util::unique_function_nonser<
            void(
                error_code const&
              , parcelset::locality const&
              , boost::shared_ptr<sender_connection>
            )
        > postprocess_handler_;

        header header_;

        MPI_Request request_;
        MPI_Request *request_ptr_;
        std::size_t chunks_idx_;
        char ack_;

        performance_counters::parcels::gatherer & parcels_sent_;

        parcelset::locality there_;
    };
}}}}

#endif

