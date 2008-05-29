//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM)
#define HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/asio.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/naming/locality.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/parcelset/parcel.hpp>
#include <hpx/parcelset/server/parcelport_queue.hpp>
#include <hpx/parcelset/server/parcelport_server_connection.hpp>
#include <hpx/parcelset/parcelport_connection.hpp>
#include <hpx/parcelset/connection_cache.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    /// The parcelport is the lowest possible representation of the parcelset
    /// inside a locality. It provides the minimal functionality to send and
    /// to receive parcels.
    class parcelport : boost::noncopyable
    {
    private:
        static void default_write_handler(boost::system::error_code const&, 
            std::size_t) {}
        
        // avoid warnings about using \a this in member initializer list
        parcelport& This() { return *this; }
        
    public:
        /// Construct the parcelport server to listen on the specified TCP 
        /// address and port, and serve up requests to the parcelport.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 incoming requests
        /// \param address  [in] The name (IP address) this instance should
        ///                 listen at.
        /// \param port     [in] The (IP) port this instance should listen at.
        explicit parcelport(util::io_service_pool& io_service_pool, 
            std::string address = "localhost", unsigned short port = HPX_PORT);

        /// Construct the parcelport server to listen to the endpoint given by 
        /// the locality and serve up requests to the parcelport.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve 
        ///                 incoming requests
        /// \param here     [in] The locality this instance should listen at.
        parcelport(util::io_service_pool& io_service_pool, 
            naming::locality here);

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

        /// The get_parcel command returns a parcel, or if the parcel set is 
        /// empty then false is returned. 
        ///
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the next parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no parcel is available in the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
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
        bool register_event_handler(F sink)
        {
            return parcels_.register_event_handler(sink);
        }

        template <typename F, typename Connection>
        bool register_event_handler(F sink, Connection& conn)
        {
            return parcels_.register_event_handler(sink, conn);
        }

        /// Allow access to locality instance.
        ///
        /// This accessor returns a reference to the locality this parcelport
        /// is associated with.
        naming::locality const& here() const
        {
            return here_;
        }
        
    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e);
        void handle_read_completion(boost::system::error_code const& e);

        /// send the parcel to the specified address
        template <typename Handler>
        void send_parcel(parcel const& p, naming::address const& addr, Handler f)
        {
            parcelport_connection_ptr client_connection(
                connection_cache_.get(addr.locality_.get_endpoint()));
                
            if (!client_connection) {
            // Start an asynchronous connect operation. The parcel gets 
            // serialized inside the connection constructor, no need to keep 
            // the original parcel alive after this call returned.
                client_connection.reset(new parcelport_connection(
                        io_service_pool_.get_io_service(), 
                        addr.locality_.get_endpoint(), connection_cache_)); 
                client_connection->set_parcel(p);
                
//             std::cerr << addr.locality_.get_endpoint().address().to_string() << ":" 
//                       << addr.locality_.get_endpoint().port()
//                       << std::endl;
            
                client_connection->socket().async_connect(
                    addr.locality_.get_endpoint(),
                    boost::bind(&parcelport::handle_connect<Handler>, this,
                        boost::asio::placeholders::error, client_connection, f));
            }
            else {
            // reuse an existing connection
                client_connection->set_parcel(p);
                client_connection->async_write(f);
            }
        }
        
        // helper functions for sending parcels
        template <typename Handler>
        void handle_connect(boost::system::error_code const& e,
            parcelport_connection_ptr conn, Handler f)
        {
            if (!e) {
                // connected successfully, now transmit the data
                conn->async_write(f);
            }
            else {
                f(e, 0);     // report the error back to the user
            }
        }

    private:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool& io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor acceptor_;

        /// The handler for all incoming requests.
        server::parcelport_queue parcels_;

        /// The next connection to be accepted.
        server::parcelport_connection_ptr new_server_connection_;
        
        /// The connection cache for sending connections
        connection_cache connection_cache_;
        
        /// The local locality
        naming::locality here_;
    };

///////////////////////////////////////////////////////////////////////////////
}}
    
#endif
