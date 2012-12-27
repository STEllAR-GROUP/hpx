//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_PARCELPORT_NOV_25_2012_0425PM)
#define HPX_PARCELSET_SHMEM_PARCELPORT_NOV_25_2012_0425PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/shmem/data_buffer_cache.hpp>
#include <hpx/runtime/parcelset/shmem/data_window.hpp>
#include <hpx/runtime/parcelset/shmem/acceptor.hpp>
#include <hpx/runtime/parcelset/shmem/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/server/shmem/parcelport_server_connection.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/util/connection_cache.hpp>

#include <boost/cstdint.hpp>

#include <set>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    /// The parcelport is the lowest possible representation of the parcelset
    /// inside a locality. It provides the minimal functionality to send and
    /// to receive parcels. This parcelport manages connections over shared
    /// memory.
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
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        /// \param f        [in] A function object to be invoked on successful
        ///                 completion or on errors. The signature of this
        ///                 function object is expected to be:
        ///
        /// \code
        ///      void handler(boost::system::error_code const& err,
        ///                   std::size_t bytes_written);
        /// \endcode
        void put_parcel(parcel const & p, write_handler_type f);

        /// Cache specific functionality
        void remove_from_connection_cache(naming::locality const& loc)
        {
            connection_cache_.clear(loc);
        }

        /// Retrieve the type of the locality represented by this parcelport
        connection_type get_type() const
        {
            return connection_shmem;
        }

        /// Return the thread pool if the name matches
        util::io_service_pool* get_thread_pool(char const* name);

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e,
            server::shmem::parcelport_connection_ptr);
        void handle_read_completion(boost::system::error_code const& e,
            server::shmem::parcelport_connection_ptr);

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

    private:
        /// The pool of io_service objects used to perform asynchronous operations.
        util::io_service_pool io_service_pool_;

        /// Acceptor used to listen for incoming connections.
        acceptor* acceptor_;
        std::size_t connection_count_;

        /// The connection cache for sending connections
        util::connection_cache<parcelport_connection, naming::locality> connection_cache_;

        /// The cache holding data_buffers
        data_buffer_cache data_buffer_cache_;

        /// The list of accepted connections
        typedef std::set<server::shmem::parcelport_connection_ptr> accepted_connections_set;
        accepted_connections_set accepted_connections_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
