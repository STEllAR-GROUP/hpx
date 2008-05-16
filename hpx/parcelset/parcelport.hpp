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
        
        // avoid warnings about using 'this' in member initializer list
        parcelport& This() { return *this; }
        
    public:
        /// Construct the parcelport server to listen on the specified TCP 
        /// address and port, and serve up requests to the parcelport.
        ///
        /// resolver        [in] A reference to the DGAS client to use for 
        ///                 address translation requests to be made by the 
        ///                 parcelport.
        /// address         [in] The name (IP address) this instance should
        ///                 listen at.
        /// port            [in] The (IP) port this instance should listen at.
        /// io_service_pool_size
        ///                 [in] The number of threads to run to serve incoming
        ///                 requests
        explicit parcelport(naming::resolver_client& resolver, 
            std::string address = "localhost", unsigned short port = HPX_PORT, 
            std::size_t io_service_pool_size = 1);

        /// Construct the parcelport server to listen to the endpoint given by 
        /// the locality and serve up requests to the parcelport.
        ///
        /// resolver        [in] A reference to the DGAS client to use for 
        ///                 address translation requests to be made by the 
        ///                 parcelport.
        /// here            [in] The locality this instance should listen at.
        /// io_service_pool_size
        ///                 [in] The number of threads to run to serve incoming
        ///                 requests
        parcelport(naming::resolver_client& resolver, naming::locality here, 
            std::size_t io_service_pool_size = 1);

        /// Start the parcelport threads, enabling the parcel receiver
        ///
        /// blocking        [in] If blocking is set to 'true' the routine will 
        ///                 not return before stop() has been called, otherwise
        ///                 the routine returns immediately.
        bool run (bool blocking = true);
        
        /// Stop the io_service's loop and wait for all threads to join
        void stop();

        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        ///
        /// The function sync_put_parcel() is synchronous.
        ///
        /// p               [in, out] A reference to the parcel to send. The 
        ///                 function does not return before the parcel has been
        ///                 transmitted. The parcel 'p' will be modified in 
        ///                 place, as it will get set the resolved destination
        ///                 address and parcel id (if not already set).
        parcel_id sync_put_parcel(parcel& p);
        
        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        //
        /// The function put_parcel() is asynchronous, the provided functor 
        /// gets invoked on completion of the send operation or on any error.
        ///
        /// Note: the parcel must be kept alive in user land for the whole 
        ///       operation, no internal copies are made
        ///
        /// p               [in, out] A reference to the parcel to send. The 
        ///                 parcel 'p' will be modified in place, as it will 
        ///                 get set the resolved destination address and parcel 
        ///                 id (if not already set).
        /// f               [in] A function object to be invoked on successful
        ///                 completion or on errors. The signature of this
        ///                 function object is expected to be:
        ///
        ///                     void f (boost::system::error_code const& err, 
        ///                             std::size_t );
        ///
        ///                 where 'err'  is the status code of the operation and
        ///                       'size' is the number of successfully 
        ///                              transferred bytes.
        template <typename Handler>
        parcel_id put_parcel(parcel& p, Handler f)
        {
            // ensure parcel id is set
            if (!p.get_parcel_id())
                p.set_parcel_id(id_range_.get_id());

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

        /// This put_parcel() function overload is asynchronous, but no 
        /// callback functor is provided by the user. 
        ///
        /// Note: the parcel must be kept alive in user land for the whole 
        ///       operation, no internal copies are made
        ///
        /// p               [in, out] A reference to the parcel to send. The 
        ///                 parcel 'p' will be modified in place, as it will 
        ///                 get set the resolved destination address and parcel 
        ///                 id (if not already set).
        parcel_id put_parcel(parcel& p)
        {
            return put_parcel(p, &parcelport::default_write_handler);
        }

        /// The get_parcel command returns a parcel, or if the parcel set is 
        /// empty then false is returned. 
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelport.
        ///
        /// p               [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns 'true' 
        ///                 this will be the next received parcel.
        ///
        /// returns         'true' if the next parcel has been retrieved 
        ///                 successfully. 
        ///                 'false' if no parcel is available in the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(parcel& p)
        {
            return parcels_.get_parcel(p);
        }

        /// Parcels may be typed by the action class they wish to invoke. Return
        /// next parcel of given action class (FIXME: shouldn't we add the 
        /// component type here as well?)
        
        /// This get_parcel() overload returns the next available parcel 
        /// addressed to any instance of a component of type 'c'.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelport.
        ///
        /// c               [in] The component type the parcel has to be 
        ///                 addressed to.
        /// p               [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns 'true' 
        ///                 this will be the next received parcel.
        ///
        /// returns         'true' if the next parcel has been retrieved 
        ///                 successfully. 
        ///                 'false' if no corresponding parcel is available in 
        ///                 the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(components::component_type c, parcel& p)
        {
            return parcels_.get_parcel(c, p);
        }
        
        /// This get_parcel() overload returns the parcel with the given parcel 
        /// tag (id).
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelport.
        ///
        /// tag             [in] The parcel tag (id) of the parcel to retrieve.
        /// p               [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns 'true' 
        ///                 this will be the next received parcel.
        ///
        /// returns         'true' if the parcel has been retrieved 
        ///                 successfully. 
        ///                 'false' if no corresponding parcel is available in 
        ///                 the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(parcel_id tag, parcel& p)
        {
            return parcels_.get_parcel(tag, p);
        }
        
        /// This get_parcel() overload returns the parcel being sent from the 
        /// locality with the given source id.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelport.
        ///
        /// source          [in] The id of the source locality.
        /// p               [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns 'true' 
        ///                 this will be the next received parcel.
        ///
        /// returns         'true' if the parcel has been retrieved 
        ///                 successfully. 
        ///                 'false' if no corresponding parcel is available in 
        ///                 the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel_from(naming::id_type source, parcel& p)
        {
            return parcels_.get_parcel_from(source, p);
        }
        
        /// This get_parcel() overload returns the parcel being to the given
        /// destination address.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelport.
        ///
        /// dest            [in] The id of the destination component.
        /// p               [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns 'true' 
        ///                 this will be the next received parcel.
        ///
        /// returns         'true' if the parcel has been retrieved 
        ///                 successfully. 
        ///                 'false' if no corresponding parcel is available in 
        ///                 the parcelport
        ///
        /// The returned parcel will be no longer available from the parcelport
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel_for(naming::id_type dest, parcel& p)
        {
            return parcels_.get_parcel_for(dest, p);
        }
        
        /// Register an event handler to be called whenever a parcel has been 
        /// received
        ///
        /// sink            [in] A function object to be invoked whenever a 
        ///                 parcel has been received by the parcelport. It is 
        ///                 possible to register more than one (different) 
        ///                 function object. The signature of this function 
        ///                 object is expected to be:
        ///
        ///                     void sink(hpx::parcelset::parcelport& pp);
        ///
        ///                 where 'pp' is a reference to the parcelport this
        ///                 function object instance is invoked by.
        template <typename F>
        bool register_event_handler(F sink)
        {
            return parcels_.register_event_handler(sink);
        }

        /// Allow access to DGAS resolver instance. 
        ///
        /// This accessor returns a reference to the DGAS resolver client object
        /// the parcelport has been initialized with (see parcelport 
        /// constructors).
        naming::resolver_client& get_resolver()
        {
            return resolver_;
        }
        
        /// Allow access to locality instance.
        ///
        /// This accessor returns a reference to the locality this parcelport
        /// is associated with.
        naming::locality const& here() const
        {
            return here_;
        }
        
        /// Return the prefix of this locality
        ///
        /// This accessor allows to retrieve the prefix value being assigned to 
        /// the locality this parcelport is associated with. This returns the
        /// same value as would be returned by:
        ///
        ///     naming::id_type prefix;
        ///     get_resolver().get_prefix(here, prefix);
        /// 
        /// but doesn't require the fully DGAS round trip as the prefix value 
        /// is cached inside the parcelport.
        naming::id_type const& get_prefix() const 
        { 
            return prefix_; 
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
            boost::asio::io_service& ios = io_service_pool_.get_io_service();
            server::connection_ptr client_connection (
                new server::connection(ios));

            using boost::asio::ip::tcp;
            tcp::endpoint const& locality_endpoint = addr.locality_.get_endpoint();
            std::string dest /*("localhost");*/ (locality_endpoint.address().to_string());
            std::string port = boost::lexical_cast<std::string>(locality_endpoint.port());
            
            tcp::resolver resolver(ios); //acceptor_.io_service());
            tcp::resolver::query query(dest, port);
            tcp::endpoint endpoint = *resolver.resolve(query);

            std::cerr << endpoint.address().to_string() << ":" 
                      << endpoint.port() << std::endl
                      << addr.locality_.get_endpoint().address().to_string() << ":" 
                      << addr.locality_.get_endpoint().port()
                      << std::endl;
            
            client_connection->socket().async_connect(
                endpoint, //addr.locality_.get_endpoint(),
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

        // generate next unique id
        parcel_id get_next_id()
        {
            return id_range_.get_id();
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
        
        /// The DGAS client
        naming::resolver_client& resolver_;
        
        /// The local locality
        naming::locality here_;
        
        /// The site prefix of the locality 'here_'
        naming::id_type prefix_;

        /// The site current range of ids to be used for id_type instances
        util::unique_ids id_range_;
    };

///////////////////////////////////////////////////////////////////////////////
}}
    
#endif
