//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM)
#define HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/server/parcelport_server_connection.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/util.hpp>
#include <hpx/util/logging.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    /// The parcelport is the lowest possible representation of the parcelset
    /// inside a locality. It provides the minimal functionality to send and
    /// to receive parcels.
    class HPX_EXPORT parcelport : boost::noncopyable
    {
    private:
        // avoid warnings about using \a this in member initializer list
        parcelport& This() { return *this; }

    public:
        /// Construct the parcelport server to listen to the endpoint given by 
        /// the locality and serve up requests to the parcelport.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 incoming requests
        /// \param here     [in] The locality this instance should listen at.
        parcelport(util::io_service_pool& io_service_pool, 
            naming::locality here);

        ~parcelport();

        /// Start the parcelport threads, enabling the parcel receiver
        ///
        /// \param blocking [in] If blocking is set to \a true the routine will 
        ///                 not return before stop() has been called, otherwise
        ///                 the routine returns immediately.
        bool run (bool blocking = true);

        /// Stop the io_service's loop and wait for all threads to join
        ///
        /// \param blocking [in] If blocking is set to \a false the routine will 
        ///                 return immediately, otherwise it will wait for all
        ///                 worker threads to exit.
        void stop (bool blocking = true);

        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        //
        /// \note The function put_parcel() is asynchronous, the provided   
        /// function or function object gets invoked on completion of the send 
        /// operation or on any error.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The 
        ///                 parcel \a p will be modified in place, as it will 
        ///                 get set the resolved destination address and parcel 
        ///                 id (if not already set).
        /// \param f        [in] A function object to be invoked on successful
        ///                 completion or on errors. The signature of this
        ///                 function object is expected to be:
        ///
        ///                     void f (boost::system::error_code const& err, 
        ///                             std::size_t );
        ///
        ///                 where \a err is the status code of the operation and
        ///                       \a size is the number of successfully 
        ///                              transferred bytes.
        template <typename Handler>
        parcel_id put_parcel(parcel& p, Handler f)
        {
            // send the parcel to its destination
            send_parcel(p, p.get_destination_addr(), f);

            // return parcel id of the parcel being sent
            return p.get_parcel_id();
        }

        // The get_parcel command returns a parcel, or if the parcel set is 
        // empty then false is returned. 
        //
        // \param p        [out] The parcel instance to be filled with the 
        //                 received parcel. If the functioned returns \a true 
        //                 this will be the next received parcel.
        //
        // \returns        \a true if the next parcel has been retrieved 
        //                 successfully. 
        //                 \a false if no parcel is available in the parcelport
        //
        // The returned parcel will be no longer available from the parcelport
        // as it is removed from the internal queue of received parcels.
//         bool get_parcel(parcel& p)
//         {
//             return parcels_.get_parcel(p);
//         }

        /// Register an event handler to be called whenever a parcel has been 
        /// received
        ///
        /// \param sink     [in] A function object to be invoked whenever a 
        ///                 parcel has been received by the parcelport. It is 
        ///                 possible to register more than one (different) 
        ///                 function object. The signature of this function 
        ///                 object is expected to be:
        ///
        /// \code
        ///      void sink (hpx::parcelset::parcelport& pp
        ///                 hpx::parcelset::parcel const& p);
        /// \endcode
        ///
        ///                 where \a pp is a reference to the parcelport this
        ///                 function object instance is invoked by, and \a dest
        ///                 is the local destination address of the parcel.
        template <typename F>
        void register_event_handler(F sink)
        {
            parcels_.register_event_handler(sink);
        }

        /// \brief Allow access to the locality this parcelport is associated 
        /// with.
        ///
        /// This accessor returns a reference to the locality this parcelport
        /// is associated with.
        naming::locality const& here() const
        {
            return here_;
        }

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e,
            server::parcelport_connection_ptr);
        void handle_read_completion(boost::system::error_code const& e);

        /// send the parcel to the specified address
        template <typename Handler>
        void send_parcel(parcel const& p, naming::address const& addr, Handler f)
        {
            parcelport_connection_ptr client_connection(
                connection_cache_.get(addr.locality_));

            if (!client_connection) {
//                 LPT_(info) << "parcelport: creating new connection to: " 
//                            << addr.locality_;

            // The parcel gets serialized inside the connection constructor, no 
            // need to keep the original parcel alive after this call returned.
                client_connection.reset(new parcelport_connection(
                        io_service_pool_.get_io_service(), addr.locality_, 
                        connection_cache_)); 
                client_connection->set_parcel(p);

            // connect to the target locality, retry if needed
                boost::system::error_code error = boost::asio::error::try_again;
                for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
                {
                    try {
                        naming::locality::iterator_type end = addr.locality_.connect_end();
                        for (naming::locality::iterator_type it = 
                                addr.locality_.connect_begin(io_service_pool_.get_io_service()); 
                             it != end; ++it)
                        {
                            client_connection->socket().close();
                            client_connection->socket().connect(*it, error);
                            if (!error) 
                                break;
                        }
                        if (!error) 
                            break;

                        // we wait for a really short amount of time (usually 100µs)
                        boost::this_thread::sleep(boost::get_system_time() + 
                            boost::posix_time::microseconds(HPX_NETWORK_RETRIES_SLEEP));
                    }
                    catch (boost::system::error_code const& e) {
                        HPX_THROW_EXCEPTION(network_error, 
                            "parcelport::send_parcel", e.message());
                    }
                }
                if (error) {
                    client_connection->socket().close();

                    HPX_OSSTREAM strm;
                    strm << error.message() << " (while trying to connect to: " 
                         << addr.locality_ << ")";
                    HPX_THROW_EXCEPTION(network_error, 
                        "parcelport::send_parcel", 
                        HPX_OSSTREAM_GETSTRING(strm));
                }

            // Start an asynchronous write operation.
                client_connection->async_write(f);
            }
            else {
//                 LPT_(info) << "parcelport: reusing existing connection to: " 
//                            << addr.locality_;

            // reuse an existing connection
                client_connection->set_parcel(p);
                client_connection->async_write(f);
            }
        }

    private:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool& io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor* acceptor_;

        /// The handler for all incoming requests.
        server::parcelport_queue parcels_;

        /// The connection cache for sending connections
        util::connection_cache<parcelport_connection> connection_cache_;

        /// The local locality
        naming::locality here_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
