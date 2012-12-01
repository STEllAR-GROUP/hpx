//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_PARCELPORT_CONNECTION_NOV_25_2012_0916PM)
#define HPX_PARCELSET_SHMEM_PARCELPORT_CONNECTION_NOV_25_2012_0916PM

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/shmem/data_window.hpp>
#include <hpx/runtime/parcelset/shmem/data_buffer.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

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
namespace hpx { namespace parcelset { namespace shmem
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a sending parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
            naming::locality const& here, naming::locality const& there,
//             util::connection_cache<parcelport_connection, naming::locality>& cache,
            performance_counters::parcels::gatherer& parcels_sent);

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            window_.shutdown(ec); // shut down data connection
            window_.close(ec);    // close the socket to give it back to the OS
        }

        void set_parcel (parcel const& p)
        {
            set_parcel(std::vector<parcel>(1, p));
        }

        void set_parcel (std::vector<parcel> const& p);

        /// Get the window associated with the parcelport_connection.
        parcelset::shmem::data_window& window() { return window_; }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            /// Increment sends and begin timer.
            send_data_.time_ = timer_.elapsed_nanoseconds();

            // Write the serialized data to the socket. 
            //
            // this additional wrapping of the handler into a bind object is
            // needed to keep  this parcelport_connection object alive for the whole
            // write operation
            void (parcelport_connection::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler, ParcelPostprocess>)
                = &parcelport_connection::handle_write<Handler, ParcelPostprocess>;

            window_.async_write(out_buffer_,
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
            // just call initial handler
            boost::get<0>(handler)(e, bytes);

            // complete data point and push back onto gatherer
            send_data_.time_ = timer_.elapsed_nanoseconds() - send_data_.time_;
            parcels_sent_.add_data(send_data_);

            // now handle the acknowledgment byte which is sent by the receiver
            void (parcelport_connection::*f)(boost::tuple<Handler, ParcelPostprocess>)
                = &parcelport_connection::handle_read_ack<Handler, ParcelPostprocess>;

            window_.async_read_ack(boost::bind(f, shared_from_this(), handler));
        }

        template <typename Handler, typename ParcelPostprocess>
        void handle_read_ack(boost::tuple<Handler, ParcelPostprocess> handler)
        {
            // now we can give this connection back to the cache
            out_buffer_.close();

            send_data_.bytes_ = 0;
            send_data_.time_ = 0;
            send_data_.serialization_time_ = 0;
            send_data_.num_parcels_ = 0;

            // Call post-processing handler, which will send remaining pending
            // parcels. Pass along the connection so it can be reused if more
            // parcels have to be sent.
            boost::get<1>(handler)(there_, shared_from_this());
        }

    private:
        /// Window for the parcelport_connection.
        parcelset::shmem::data_window window_;

        /// buffer for outgoing data
        data_buffer out_buffer_;

        /// the other (receiving) end of this connection
        naming::locality there_;

        /// The connection cache for sending connections
//         util::connection_cache<parcelport_connection, naming::locality>& connection_cache_;

        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point send_data_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    ///////////////////////////////////////////////////////////////////////////
    inline std::string full_endpoint_name(hpx::naming::locality const& l)
    {
        return l.get_address() + "." +
            boost::lexical_cast<std::string>(l.get_port());
    }
}}}

#endif
