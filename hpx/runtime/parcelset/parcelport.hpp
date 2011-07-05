//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
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
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/connection_cache.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/stringstream.hpp>
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

        typedef boost::function<
              void(boost::system::error_code const&, std::size_t)
        > handler_type;
    
        void put_parcel(parcel& p, handler_type f)
        {
            // send the parcel to its destination
            send_parcel(p, p.get_destination_addr(), f);
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

        util::io_service_pool& get_io_service_pool()
        {
            return io_service_pool_;
        }

        util::connection_cache<parcelport_connection>& get_connection_cache()
        {
            return connection_cache_;
        }

        util::unique_ids& get_id_range()
        {
            return id_range_;
        }

        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper
        ) {
            id_range_.set_range(lower, upper);
        }

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e,
            server::parcelport_connection_ptr);
        void handle_read_completion(boost::system::error_code const& e);

        /// send the parcel to the specified address
        void send_parcel(parcel const& p, naming::address const& addr, 
            handler_type f);

    private:
        /// The site current range of ids to be used for id_type instances
        util::unique_ids id_range_;

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
