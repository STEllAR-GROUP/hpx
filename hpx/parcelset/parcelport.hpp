//  Copyright (c) 2007-2008 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
// 
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <hpx/naming/name.hpp>
#include <hpx/naming/address.hpp>
#include <hpx/naming/locality.hpp>
#include <hpx/naming/resolver_client.hpp>
#include <hpx/util/generate_unique_ids.hpp>

#include <hpx/util/io_service_pool.hpp>
#include <hpx/parcelset/parcel.hpp>
#include <hpx/parcelset/server/parcel_queue.hpp>
#include <hpx/parcelset/server/parcel_connection.hpp>

#if HPX_USE_TBB != 0
#include <tbb/atomic.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    class parcelport : boost::noncopyable
    {
        static void default_write_handler(boost::system::error_code const&, 
            std::size_t) {}
        
        // avoid warning about using this in member initializer list
        parcelport const& This() const { return *this; }
        
    public:
        /// Construct the server to listen on the specified TCP address and port, 
        /// and serve up requests to the address translation service.
        explicit parcelport(naming::resolver_client& resolver, 
            std::string address = "localhost", unsigned short port = HPX_PORT, 
            std::size_t io_service_pool_size = 1);

        /// Construct the server to listen to the endpoint given by the 
        /// locality and serve up requests to the address translation service.
        parcelport(naming::resolver_client& resolver, naming::locality here, 
            std::size_t io_service_pool_size = 1);

        void run (bool blocking = true);
        
        /// Stop the io_service's loop and wait for all threads to join
        void stop();

        ///////////////////////////////////////////////////////////////////////
        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        /// This function is synchronous.
        parcel_id sync_put_parcel(parcel const& p);
        
        ///////////////////////////////////////////////////////////////////////
        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        /// This function is asynchronous, the provided functor gets invoked on
        /// completion of the send operation or on any error.
        /// Note: the parcel must be kept alive in user land for the whole 
        ///       operation, not copies are made
        template <typename Handler>
        parcel_id put_parcel(parcel const& p, Handler f)
        {
            // ensure parcel id is set
            if (!p.get_parcel_id())
                p.set_parcel_id(get_next_parcel_id());

            // ensure the source locality id is set (if no component id is given)
            if (!p.get_source())
                p.set_source(prefix_);
                
            // resolve destination address, if needed
            if (!p.get_destination_addr()) {
                naming::address addr;
                if (!resolver_.resolve(p.get_destination(), addr)) {
                    throw exception(unknown_component_address, 
                        "Unknown destination address");
                }
                p.set_destination_addr(addr);
            }
            
            // send the parcel to its destination
            send_parcel(p, p.get_destination_addr(), f);
            
            // return parcel id of the parcel being sent
            return p.get_parcel_id();
        }

        /// This function is asynchronous, no callback functor is provided
        parcel_id put_parcel(parcel const& p)
        {
            return put_parcel(p, &parcelport::default_write_handler);
        }

        ///////////////////////////////////////////////////////////////////////////
        /// The get_parcel command returns a parcel, or if the parcel set is 
        /// empty then false is returned. 
        
        /// return 'next' available parcel
        bool get_parcel(parcel& p)
        {
            return parcels_.get_parcel(p);
        }

        /// Parcels may be typed by the action class they wish to invoke. Return
        /// next parcel of given action class (FIXME: shouldn't we add the 
        /// component type here as well?)
        bool get_parcel(components::component_type c, parcel& p)
        {
            return parcels_.get_parcel(c, p);
        }
        
        /// Return the parcel with the given parcel tag
        bool get_parcel(parcel_id tag, parcel& p)
        {
            return parcels_.get_parcel(tag, p);
        }
        
        /// Return the next parcel received from the given source locality
        bool get_parcel_from(naming::id_type source, parcel& p)
        {
            return parcels_.get_parcel_from(source, p);
        }
        
        /// Return the next parcel for the given destination address
        bool get_parcel_for(naming::id_type dest, parcel& p)
        {
            return parcels_.get_parcel_for(dest, p);
        }
        
        /// Return the prefix of this locality
        boost::uint64_t get_prefix() const { return prefix_; }
        
        /// register an event handler to be called whenever a parcel has been 
        /// received
        template <typename F>
        bool register_event_handler(F sink)
        {
            return parcels_.register_event_handler(sink);
        }

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e);
        void handle_read_completion(boost::system::error_code const& e);

        /// send the parcel to the specified address
        template <typename Handler>
        void send_parcel(parcel const& p, naming::address const& addr, Handler f)
        {
            // Start an asynchronous connect operation.
            server::connection_ptr client_connection (
                new server::connection(io_service_pool_.get_io_service()));

            client_connection->socket().async_connect(
                addr.locality_.get_endpoint(),
                boost::bind(&parcelport::handle_connect<Handler>, this,
                    boost::asio::placeholders::error, client_connection, 
                    boost::ref(p), f));
        }
        
        // helper functions for sending parcels
        template <typename Handler>
        void handle_connect(boost::system::error_code const& e,
            server::connection_ptr conn, parcel const& p, Handler f)
        {
            if (!e) {
                // connected successfully, now transmit the data
                conn->async_write(p, f);
            }
            else {
                f(e, 0);     // report the error back to the user
            }
        }

        // generate next unique parcel id
        parcel_id get_next_parcel_id()
        {
            return util::unique_ids::instance->get_id();
        }

    public:
        // allow access to resolver instance
        naming::resolver_client& get_resolver()
        {
            return resolver_;
        }
        
        // allow access to locality instance
        naming::locality const& here() const
        {
            return here_;
        }
        
    private:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        boost::asio::ip::tcp::acceptor acceptor_;

        /// The handler for all incoming requests.
        server::parcel_queue parcels_;

        /// The next connection to be accepted.
        server::connection_ptr new_server_connection_;
        
        /// The GAS client
        naming::resolver_client& resolver_;
        
        /// The local locality
        naming::locality here_;
        
        /// The site prefix to be used for id_type instances
        boost::uint64_t prefix_;
    };

///////////////////////////////////////////////////////////////////////////////
}}
    
#endif
