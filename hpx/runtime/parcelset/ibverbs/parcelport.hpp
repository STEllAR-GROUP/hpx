//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_IBVERBS_PARCELPORT_HPP)
#define HPX_PARCELSET_IBVERBS_PARCELPORT_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/server/ibverbs/parcelport_server_connection.hpp>
#include <hpx/runtime/parcelset/ibverbs/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/ibverbs/acceptor.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>

#include <boost/cstdint.hpp>

#include <set>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace ibverbs
{
    /// The parcelport is the lowest possible representation of the parcelset
    /// inside a locality. It provides the minimal functionality to send and
    /// to receive parcels. This parcelport manages connections over infiniband
    class HPX_EXPORT parcelport : public parcelset::parcelport
    {
    public:
        /// Construct the parcelport on the given locality.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve
        ///                 incoming requests
        /// \param here     [in] The locality this instance should listen at.
        parcelport(util::runtime_configuration const& ini,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread);

        ~parcelport();

        /// Start the parcelport I/O thread pool.
        ///
        /// \param blocking [in] If blocking is set to \a true the routine will
        ///                 not return before stop() has been called, otherwise
        ///                 the routine returns immediately.
        bool run(bool blocking = true);

        /// Stop the parcelport I/O thread pool.
        ///
        /// \param blocking [in] If blocking is set to \a false the routine will
        ///                 return immediately, otherwise it will wait for all
        ///                 worker threads to exit.
        void stop(bool blocking = true);

        /// Queues a parcel for transmission to another locality
        ///
        /// \note The function put_parcel() is asynchronous, the provided
        /// function or function object gets invoked on completion of the send
        /// operation or on any error.
        ///
        /// \param p        [in] A reference to the parcel to send.
        /// \param f        [in] A function object to be invoked on successful
        ///                 completion or on errors. The signature of this
        ///                 function object is expected to be:
        ///
        /// \code
        ///      void handler(boost::system::error_code const& err,
        ///                   std::size_t bytes_written);
        /// \endcode
        void put_parcel(parcel const & p, write_handler_type f);

        /// Queues a list of parcels for transmission to another locality
        ///
        /// \note The function put_parcels() is asynchronous, the provided
        /// functions or function objects get invoked on completion of the send
        /// operation or on any error.
        ///
        /// \param parcels  [in] A reference to the list of parcels to send.
        /// \param handlers [in] A list of function objects to be invoked on
        ///                 successful completion or on errors. The signature of
        ///                 these function objects is expected to be:
        ///
        /// \code
        ///      void handler(boost::system::error_code const& err,
        ///                   std::size_t bytes_written);
        /// \endcode
        void put_parcels(std::vector<parcel> const & parcels,
            std::vector<write_handler_type> const& handlers);


        /// Retrieve the type of the locality represented by this parcelport
        connection_type get_type() const
        {
            return connection_ibverbs;
        }
        
        /// Cache specific functionality
        void remove_from_connection_cache(naming::locality const& loc)
        {
            connection_cache_.clear(loc);
        }

        /// Return the thread pool if the name matches
        util::io_service_pool* get_thread_pool(char const* name);

        /// Return the given connection cache statistic
        boost::int64_t get_connection_cache_statistics(
            connection_cache_statistics_type t, bool reset);

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e,
            server::ibverbs::parcelport_connection_ptr);
        void handle_read_completion(boost::system::error_code const& e,
            server::ibverbs::parcelport_connection_ptr);

        /// helper function to send remaining pending parcels
        void send_pending_parcels_trampoline(
            boost::system::error_code const& ec,
            naming::locality const& prefix,
            parcelport_connection_ptr client_connection);
        void send_pending_parcels(parcelport_connection_ptr client_connection,
            std::vector<parcel> const&, std::vector<write_handler_type> const&);

        /// \brief Retrieve a new connection
        parcelport_connection_ptr get_connection(naming::locality const& l,
            error_code& ec = throws);
        parcelport_connection_ptr get_connection_wait(naming::locality const& l,
            error_code& ec = throws);

        parcelport_connection_ptr get_connection(naming::locality const& l,
            parcelport_connection_ptr client_connection, error_code& ec = throws);

        void send_parcels_or_reclaim_connection(naming::locality const& locality_id,
            parcelport_connection_ptr const& client_connection);

    private:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        acceptor* acceptor_;

        /// The connection cache for sending connections
        util::connection_cache<parcelport_connection, naming::locality> connection_cache_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
