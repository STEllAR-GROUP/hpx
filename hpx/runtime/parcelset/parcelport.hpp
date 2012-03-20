//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007      Richard D Guidry Jr
//  Copyright (c) 2011      Bryce Lelbach & Katelyn Kufahl
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
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/cstdint.hpp>
#include <boost/noncopyable.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx {

namespace agas {

struct HPX_EXPORT big_boot_barrier;

}

namespace parcelset
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
        friend struct agas::big_boot_barrier;

        typedef HPX_STD_FUNCTION<
              void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        typedef HPX_STD_FUNCTION<
            void(parcelport& pp, boost::shared_ptr<std::vector<char> > const&,
                 threads::thread_priority)
        > read_handler_type;

        /// Construct the parcelport on the given locality.
        ///
        /// \param io_service_pool
        ///                 [in] The pool of networking threads to use to serve
        ///                 incoming requests
        /// \param here     [in] The locality this instance should listen at.
        parcelport(util::io_service_pool& io_service_pool
          , naming::locality here
          , std::size_t max_connection_cache_size
          , std::size_t max_connections_per_locality);

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
        void put_parcel(parcel& p, write_handler_type f)
        {
            // send the parcel to its destination
            send_parcel(p, p.get_destination_addr(), f);
        }

        /// Register an event handler to be called whenever a parcel has been
        /// received.
        ///
        /// \param sink     [in] A function object to be invoked whenever a
        ///                 parcel has been received by the parcelport. The
        ///                 signature of this function object is expected to be:
        ///
        /// \code
        ///      void handler(hpx::parcelset::parcelport& pp,
        ///                   boost::shared_ptr<std::vector<char> > const& data,
        ///                   hpx::threads::thread_priority priority);
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

        void set_range(naming::gid_type const& lower, naming::gid_type const& upper)
        {
            id_range_.set_range(lower, upper);
        }

        /// number of messages sent
        boost::int64_t get_send_count() const
        {
            return parcels_sent_.size();
        }

        /// number of messages received
        boost::int64_t get_receive_count() const
        {
            return parcels_received_.size();
        }

        /// the total time it took for all sends, from async_write to the
        /// completion handler (nanoseconds)
        boost::int64_t get_sending_time() const
        {
            return parcels_sent_.total_time() * 1000;
        }

        /// the total time it took for all receives, from async_read to the
        /// completion handler (nanoseconds)
        boost::int64_t get_receiving_time() const
        {
            return parcels_received_.total_time() * 1000;
        }

        /// total data received (bytes)
        boost::int64_t get_data_sent() const
        {
            return parcels_sent_.total_bytes();
        }

        /// total data received (bytes)
        boost::int64_t get_data_received() const
        {
            return parcels_received_.total_bytes();
        }

        void add_received_data(
            performance_counters::parcels::data_point const& data)
        {
            parcels_received_.add_data(data);
        }

    protected:
        // helper functions for receiving parcels
        void handle_accept(boost::system::error_code const& e,
            server::parcelport_connection_ptr);
        void handle_read_completion(boost::system::error_code const& e);

        /// send the parcel to the specified address
        void send_parcel(parcel const& p, naming::address const& addr,
            write_handler_type f);

        /// helper function to send remaining pending parcels
        void send_pending_parcels_trampoline(boost::uint32_t prefix);
        void send_pending_parcels(parcelport_connection_ptr client_connection, std::vector<parcel> const & parcels, std::vector<write_handler_type> const &);

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

        /// Mutex for pending parcels
        util::spinlock mtx_;
        /// The cache for pending parcels
        typedef
            std::map<
                boost::uint32_t
              , std::pair<
                    std::vector<parcel>
                  , std::vector<write_handler_type>
                >
            > pending_parcels_map;
        pending_parcels_map pending_parcels_;

        /// The local locality
        naming::locality here_;

        /// Parcel timers and their data containers.
        util::high_resolution_timer timer_;

        performance_counters::parcels::gatherer parcels_sent_;
        performance_counters::parcels::gatherer parcels_received_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
