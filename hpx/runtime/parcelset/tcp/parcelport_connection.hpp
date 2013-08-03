//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_TCP_PARCELPORT_CONNECTION_MAY_20_2008_1132PM)
#define HPX_PARCELSET_TCP_PARCELPORT_CONNECTION_MAY_20_2008_1132PM

#include <sstream>
#include <vector>
#include <set>

#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/connection_cache.hpp>
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace tcp
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
    public:
        enum state
        {
            state_initialized,
            state_reinitialized,
            state_set_parcel,
            state_async_write,
            state_handle_write,
            state_handle_read_ack,
            state_scheduled_thread,
            state_send_pending,
            state_reclaimed
        };
#endif

    public:
        /// Construct a sending parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
            naming::locality const& locality_id,
            performance_counters::parcels::gatherer& parcels_sent,
            boost::uint64_t max_outbound_size);

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the socket
            if (socket_.is_open()) {
                boost::system::error_code ec;
                socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                socket_.close(ec);    // close the socket to give it back to the OS
            }
        }

        void set_parcel (parcel const& p)
        {
            set_parcel(std::vector<parcel>(1, p));
        }

        void set_parcel (std::vector<parcel> const& p);

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
            state_ = state_async_write;
#endif
            /// Increment sends and begin timer.
            send_data_.time_ = timer_.elapsed_nanoseconds();

            // prepare chunk data for transmission, the transmission_chunks data
            // first holds all zero-copy, then all non-zero-copy chunk infos
            out_transmission_chunks_.clear();
            out_transmission_chunks_.reserve(out_chunks_.size());

            std::size_t index = 0;
            BOOST_FOREACH(util::serialization_chunk& c, out_chunks_)
            {
                if (c.type_ == util::chunk_type_pointer) {
                    out_transmission_chunks_.push_back(
                        transmission_chunk_type(index, c.size_));
                }
                ++index;
            }

            num_chunks_ = count_chunks_type(
                    static_cast<boost::uint32_t>(out_transmission_chunks_.size()),
                    static_cast<boost::uint32_t>(out_chunks_.size() -
                        out_transmission_chunks_.size())
                );

            // Write the serialized data to the socket. We use "gather-write"
            // to send both the header and the data in a single write operation.
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&out_priority_, sizeof(out_priority_)));
            buffers.push_back(boost::asio::buffer(&out_size_, sizeof(out_size_)));
            buffers.push_back(boost::asio::buffer(&out_data_size_, sizeof(out_data_size_)));

            // add chunk description
            buffers.push_back(boost::asio::buffer(&num_chunks_, sizeof(num_chunks_)));

            if (!out_transmission_chunks_.empty()) {
                // the remaining number of chunks are non-zero-copy
                BOOST_FOREACH(util::serialization_chunk& c, out_chunks_)
                {
                    if (c.type_ == util::chunk_type_index) {
                        out_transmission_chunks_.push_back(
                            transmission_chunk_type(c.data_.index_, c.size_));
                    }
                }

                buffers.push_back(boost::asio::buffer(out_transmission_chunks_.data(),
                    out_transmission_chunks_.size()*sizeof(transmission_chunk_type)));

                // add main buffer holding data which was serialized normally
                buffers.push_back(boost::asio::buffer(out_buffer_));

                // now add chunks themselves, those hold zero-copy serialized chunks
                BOOST_FOREACH(util::serialization_chunk& c, out_chunks_)
                {
                    if (c.type_ == util::chunk_type_pointer)
                        buffers.push_back(boost::asio::buffer(c.data_.cpos_, c.size_));
                }
            }
            else {
                // add main buffer holding data which was serialized normally
                buffers.push_back(boost::asio::buffer(out_buffer_));
            }

            // this additional wrapping of the handler into a bind object is
            // needed to keep  this parcelport_connection object alive for the whole
            // write operation
            void (parcelport_connection::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler, ParcelPostprocess>)
                = &parcelport_connection::handle_write<Handler, ParcelPostprocess>;

            boost::asio::async_write(socket_, buffers,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error, ::_2,
                    boost::make_tuple(handler, parcel_postprocess)));
        }

        naming::locality const& destination() const
        {
            return there_;
        }

    protected:
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

            // complete data point and push back onto gatherer
            send_data_.time_ = timer_.elapsed_nanoseconds() - send_data_.time_;
            parcels_sent_.add_data(send_data_);

            // now we can give this connection back to the cache
            out_buffer_.clear();
            out_priority_ = 0;
            out_size_ = 0;
            out_data_size_ = 0;

            out_chunks_.clear();
            out_transmission_chunks_.clear();
            num_chunks_ = count_chunks_type(0, 0);

            send_data_.bytes_ = 0;
            send_data_.time_ = 0;
            send_data_.serialization_time_ = 0;
            send_data_.num_parcels_ = 0;

            // now handle the acknowledgement byte which is sent by the receiver
#if defined(__linux) || defined(linux) || defined(__linux__)
            boost::asio::detail::socket_option::boolean<
                IPPROTO_TCP, TCP_QUICKACK> quickack(true);
            socket_.set_option(quickack);
#endif

            void (parcelport_connection::*f)(boost::system::error_code const&,
                      boost::tuple<Handler, ParcelPostprocess>)
                = &parcelport_connection::handle_read_ack<Handler, ParcelPostprocess>;

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

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
    public:
        void set_state(state newstate)
        {
            state_ = newstate;
        }
#endif

#if defined(HPX_HAVE_SECURITY)
    protected:
        template <typename Archive>
        void serialize_certificate(Archive& archive,
            std::set<boost::uint32_t>& localities, parcel const& p);

        void create_message_suffix(naming::gid_type const& parcel_id);
#endif

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for outgoing data
        boost::integer::ulittle8_t out_priority_;
        boost::integer::ulittle64_t out_size_;
        boost::integer::ulittle64_t out_data_size_;

        std::vector<char> out_buffer_;

        typedef std::pair<boost::integer::ulittle64_t, boost::integer::ulittle64_t>
            transmission_chunk_type;
        std::vector<transmission_chunk_type> out_transmission_chunks_;

        // pair of (zero-copy, non-zero-copy) chunks
        typedef std::pair<
            boost::integer::ulittle32_t, boost::integer::ulittle32_t
        > count_chunks_type;
        count_chunks_type num_chunks_;

        std::vector<util::serialization_chunk> out_chunks_;
        boost::uint64_t max_outbound_size_;

        bool ack_;

#if defined(HPX_HAVE_SECURITY)
        bool first_message_;
#endif

        /// the other (receiving) end of this connection
        naming::locality there_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point send_data_;
        performance_counters::parcels::gatherer& parcels_sent_;

        // archive flags
        int archive_flags_;

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        state state_;
#endif
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;
#if defined(HPX_HOLDON_TO_OUTGOING_CONNECTIONS)
    typedef boost::weak_ptr<parcelport_connection> parcelport_connection_weak_ptr;
#endif
}}}

#endif
