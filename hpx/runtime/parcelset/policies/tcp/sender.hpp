//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_TCP_SENDER_HPP
#define HPX_PARCELSET_POLICIES_TCP_SENDER_HPP

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/bind/protect.hpp>
#include <boost/cstdint.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace tcp
{
    class sender
      : public parcelset::parcelport_connection<sender, std::vector<char> >
    {
    public:
        /// Construct a sending parcelport_connection with the given io_service.
        sender(boost::asio::io_service& io_service,
            naming::locality const& locality_id,
            performance_counters::parcels::gatherer& parcels_sent)
          : socket_(io_service)
          , ack_(0)
          , there_(locality_id), parcels_sent_(parcels_sent)
        {
        }

        ~sender()
        {
            // gracefully and portably shutdown the socket
            if (socket_.is_open()) {
                boost::system::error_code ec;
                socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                socket_.close(ec);    // close the socket to give it back to the OS
            }
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        naming::locality const& destination() const
        {
            return there_;
        }

        void verify(naming::locality const & parcel_locality_id) const
        {
            HPX_ASSERT(parcel_locality_id.get_address() ==
                socket_.remote_endpoint().address().to_string());
            HPX_ASSERT(parcel_locality_id.get_port() ==
                socket_.remote_endpoint().port());
        }

        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            HPX_ASSERT(buffer_);

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            state_ = state_async_write;
#endif
            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();

            // Write the serialized data to the socket. We use "gather-write"
            // to send both the header and the data in a single write operation.
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&buffer_->size_,
                sizeof(buffer_->size_)));
            buffers.push_back(boost::asio::buffer(&buffer_->data_size_,
                sizeof(buffer_->data_size_)));

            // add chunk description
            buffers.push_back(boost::asio::buffer(&buffer_->num_chunks_,
                sizeof(buffer_->num_chunks_)));

            std::vector<parcel_buffer_type::transmission_chunk_type>& chunks =
                buffer_->transmission_chunks_;
            if (!chunks.empty()) {
                buffers.push_back(
                    boost::asio::buffer(chunks.data(), chunks.size() *
                        sizeof(parcel_buffer_type::transmission_chunk_type)));

                // add main buffer holding data which was serialized normally
                buffers.push_back(boost::asio::buffer(buffer_->data_));

                // now add chunks themselves, those hold zero-copy serialized chunks
                BOOST_FOREACH(util::serialization_chunk& c, buffer_->chunks_)
                {
                    if (c.type_ == util::chunk_type_pointer)
                        buffers.push_back(boost::asio::buffer(c.data_.cpos_, c.size_));
                }
            }
            else {
                // add main buffer holding data which was serialized normally
                buffers.push_back(boost::asio::buffer(buffer_->data_));
            }

            // this additional wrapping of the handler into a bind object is
            // needed to keep  this parcelport_connection object alive for the whole
            // write operation
            void (sender::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler, ParcelPostprocess>)
                = &sender::handle_write<Handler, ParcelPostprocess>;

            boost::asio::async_write(socket_, buffers,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error, ::_2,
                    boost::make_tuple(handler, parcel_postprocess)));
        }

    private:
        /// handle completed write operation
        template <typename Handler, typename ParcelPostprocess>
        void handle_write(boost::system::error_code const& e, std::size_t bytes,
            boost::tuple<Handler, ParcelPostprocess> handler)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            state_ = state_handle_write;
#endif
            // just call initial handler
            boost::get<0>(handler)(e, bytes);
            if (e)
            {
                // inform post-processing handler of error as well
                boost::get<1>(handler)(e, there_, shared_from_this());
                return;
            }

            // complete data point and push back onto gatherer
            buffer_->data_point_.time_ =
                timer_.elapsed_nanoseconds() - buffer_->data_point_.time_;
            parcels_sent_.add_data(buffer_->data_point_);

            // now handle the acknowledgment byte which is sent by the receiver
#if defined(__linux) || defined(linux) || defined(__linux__)
            boost::asio::detail::socket_option::boolean<
                IPPROTO_TCP, TCP_QUICKACK> quickack(true);
            socket_.set_option(quickack);
#endif

            void (sender::*f)(boost::system::error_code const&,
                      boost::tuple<Handler, ParcelPostprocess>)
                = &sender::handle_read_ack<Handler, ParcelPostprocess>;

            boost::asio::async_read(socket_,
                boost::asio::buffer(&ack_, sizeof(ack_)),
                boost::bind(f, shared_from_this(), ::_1, handler));
        }

        template <typename Handler, typename ParcelPostprocess>
        void handle_read_ack(boost::system::error_code const& e,
            boost::tuple<Handler, ParcelPostprocess> handler)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            state_ = state_handle_read_ack;
#endif
            // Call post-processing handler, which will send remaining pending
            // parcels. Pass along the connection so it can be reused if more
            // parcels have to be sent.
            boost::get<1>(handler)(e, there_, shared_from_this());
        }

        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        bool ack_;

        /// the other (receiving) end of this connection
        naming::locality there_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };
}}}}

#endif
