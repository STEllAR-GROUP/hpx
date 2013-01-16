//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM)
#define HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <string>
#include <map>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace agas
{
    // forward declaration only
    struct HPX_EXPORT big_boot_barrier;
}}

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

        friend struct agas::big_boot_barrier;

    public:
        typedef HPX_STD_FUNCTION<
              void(boost::system::error_code const&, std::size_t)
        > write_handler_type;

        typedef HPX_STD_FUNCTION<
            void(parcelport& pp, boost::shared_ptr<std::vector<char> >,
                 threads::thread_priority)
        > read_handler_type;

        /// Construct the parcelport on the given locality.
        parcelport(naming::locality here)
          : parcels_(), here_(here)
        {
        }

        /// Start the parcelport I/O thread pool.
        ///
        /// \param blocking [in] If blocking is set to \a true the routine will
        ///                 not return before stop() has been called, otherwise
        ///                 the routine returns immediately.
        virtual bool run(bool blocking = true) = 0;

        /// Stop the parcelport I/O thread pool.
        ///
        /// \param blocking [in] If blocking is set to \a false the routine will
        ///                 return immediately, otherwise it will wait for all
        ///                 worker threads to exit.
        virtual void stop(bool blocking = true) = 0;

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
        virtual void put_parcel(parcel const & p, write_handler_type f) = 0;

        /// Send an early parcel through the TCP parcelport
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        virtual void send_early_parcel(parcel& p) 
        {
            BOOST_ASSERT(false);    // is implemented in tcp::parcelport only
        }

        /// Cache specific functionality
        virtual void remove_from_connection_cache(naming::locality const& loc) = 0;

        /// Retrieve the type of the locality represented by this parcelport
        virtual connection_type get_type() const = 0;

        /// Return the thread pool if the name matches
        virtual util::io_service_pool* get_thread_pool(char const* name)
        {
            return 0;     // by default no thread pool is used
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

        /// Performance counter data

        /// number of parcels sent
        std::size_t get_parcel_send_count() const
        {
            return parcels_sent_.num_parcels();
        }

        /// number of messages sent
        std::size_t get_message_send_count() const
        {
            return parcels_sent_.num_messages();
        }

        /// number of parcels received
        std::size_t get_parcel_receive_count() const
        {
            return parcels_received_.num_parcels();
        }

        /// number of messages received
        std::size_t get_message_receive_count() const
        {
            return parcels_received_.num_messages();
        }

        /// the total time it took for all sends, from async_write to the
        /// completion handler (nanoseconds)
        boost::int64_t get_sending_time() const
        {
            return parcels_sent_.total_time();
        }

        /// the total time it took for all receives, from async_read to the
        /// completion handler (nanoseconds)
        boost::int64_t get_receiving_time() const
        {
            return parcels_received_.total_time();
        }

        /// the total time it took for all sender-side serialization operations
        /// (nanoseconds)
        boost::int64_t get_sending_serialization_time() const
        {
            return parcels_sent_.total_serialization_time();
        }

        /// the total time it took for all receiver-side serialization
        /// operations (nanoseconds)
        boost::int64_t get_receiving_serialization_time() const
        {
            return parcels_received_.total_serialization_time();
        }

        /// total data sent (bytes)
        std::size_t get_data_sent() const
        {
            return parcels_sent_.total_bytes();
        }

        /// total data (type only) sent (bytes)
        std::size_t get_total_type_sent() const
        {
            return parcels_sent_.total_type_bytes();
        }

        /// total data received (bytes)
        std::size_t get_data_received() const
        {
            return parcels_received_.total_bytes();
        }

        /// total data (type only) received (bytes)
        std::size_t get_total_type_received() const
        {
            return parcels_received_.total_type_bytes();
        }

        std::size_t get_pending_parcels_count() const
        {
            util::spinlock::scoped_lock l(mtx_);
            return pending_parcels_.size();
        }

        void add_received_parcel(parcel const& p)
        {
            // do some work (notify event handlers)
            parcels_.add_parcel(p);
        }

        /// Update performance counter data
        void add_received_data(performance_counters::parcels::data_point const& data)
        {
            parcels_received_.add_data(data);
        }

        /// Create a new instance of a parcelport
        static boost::shared_ptr<parcelport> create(connection_type type,
            util::runtime_configuration const& cfg,
            HPX_STD_FUNCTION<void(std::size_t, char const*)> const& on_start_thread,
            HPX_STD_FUNCTION<void()> const& on_stop_thread);

    protected:
        /// mutex for all of the member data
        mutable util::spinlock mtx_;

        /// The handler for all incoming requests.
        server::parcelport_queue parcels_;

        /// The cache for pending parcels
        typedef std::pair<std::vector<parcel>, std::vector<write_handler_type> >
            map_second_type;
        typedef std::map<naming::locality, map_second_type> pending_parcels_map;
        pending_parcels_map pending_parcels_;

        /// The local locality
        naming::locality here_;

        /// Parcel timers and their data containers.
        performance_counters::parcels::gatherer parcels_sent_;
        performance_counters::parcels::gatherer parcels_received_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
