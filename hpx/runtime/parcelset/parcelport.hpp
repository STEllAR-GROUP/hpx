//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM)
#define HPX_PARCELSET_PARCELPORT_MAR_26_2008_1214PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/runtime/applier_fwd.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40900
#define HPX_PARCELSET_PENDING_PARCELS_WORKAROUND
#endif

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
    class HPX_EXPORT parcelport
      : public std::enable_shared_from_this<parcelport>
    {
        HPX_NON_COPYABLE(parcelport);

    private:
        // avoid warnings about using \a this in member initializer list
        parcelport& This() { return *this; }

        friend struct agas::big_boot_barrier;

    public:
        typedef util::function_nonser<
            void(boost::system::error_code const&, parcel const&)
        > write_handler_type;

        typedef util::function_nonser<
            void(parcelport& pp, std::shared_ptr<std::vector<char> >,
                 threads::thread_priority)
        > read_handler_type;

        /// Construct the parcelport on the given locality.
        parcelport(util::runtime_configuration const& ini, locality const & here,
            std::string const& type);

        /// Virtual destructor
        virtual ~parcelport() {}

        virtual bool can_bootstrap() const = 0;

        int priority() const { return priority_; }

        /// Retrieve the type of the locality represented by this parcelport
        std::string const& type() const { return type_; }

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

        /// Check if this parcelport can connect to this locality
        ///
        /// The default is to return true if it can be used at bootstrap or alternative
        /// parcelports are enabled.
        virtual bool can_connect(locality const &, bool use_alternative_parcelport)
        {
            return use_alternative_parcelport || can_bootstrap();
        }

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
        virtual void put_parcel(locality const & dest, parcel p,
            write_handler_type f) = 0;

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
        virtual void put_parcels(locality const& dests,
            std::vector<parcel> parcels,
            std::vector<write_handler_type> handlers) = 0;

        /// Send an early parcel through the TCP parcelport
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        virtual void send_early_parcel(locality const & dest, parcel p) = 0;

        /// Cache specific functionality
        virtual void remove_from_connection_cache(locality const& loc) = 0;

        /// Return the thread pool if the name matches
        virtual util::io_service_pool* get_thread_pool(char const* name) = 0;

        /// Return the given connection cache statistic
        enum connection_cache_statistics_type
        {
            connection_cache_insertions = 0,
            connection_cache_evictions = 1,
            connection_cache_hits = 2,
            connection_cache_misses = 3,
            connection_cache_reclaims = 4
        };

        // invoke pending background work
        virtual bool do_background_work(std::size_t num_thread) = 0;

        // retrieve performance counter value for given statistics type
        virtual std::int64_t get_connection_cache_statistics(
            connection_cache_statistics_type, bool reset) = 0;

        /// Return the name of this locality
        virtual std::string get_locality_name() const = 0;

        /// \brief Allow access to the locality this parcelport is associated
        /// with.
        ///
        /// This accessor returns a reference to the locality this parcelport
        /// is associated with.
        locality const& here() const
        {
            return here_;
        }

        virtual locality create_locality() const = 0;

        virtual locality agas_locality(util::runtime_configuration const& ini)
            const = 0;

        /// Performance counter data

        /// number of parcels sent
        std::int64_t get_parcel_send_count(bool reset);

        /// number of messages sent
        std::int64_t get_message_send_count(bool reset);

        /// number of parcels received
        std::int64_t get_parcel_receive_count(bool reset);

        /// number of messages received
        std::int64_t get_message_receive_count(bool reset);

        /// the total time it took for all sends, from async_write to the
        /// completion handler (nanoseconds)
        std::int64_t get_sending_time(bool reset);

        /// the total time it took for all receives, from async_read to the
        /// completion handler (nanoseconds)
        std::int64_t get_receiving_time(bool reset);

        /// the total time it took for all sender-side serialization operations
        /// (nanoseconds)
        std::int64_t get_sending_serialization_time(bool reset);

        /// the total time it took for all receiver-side serialization
        /// operations (nanoseconds)
        std::int64_t get_receiving_serialization_time(bool reset);

#if defined(HPX_HAVE_SECURITY)
        /// the total time it took for all sender-side security operations
        /// (nanoseconds)
        std::int64_t get_sending_security_time(bool reset);

        /// the total time it took for all receiver-side security
        /// operations (nanoseconds)
        std::int64_t get_receiving_security_time(bool reset);
#endif

        /// total data sent (bytes)
        std::int64_t get_data_sent(bool reset);

        /// total data (uncompressed) sent (bytes)
        std::int64_t get_raw_data_sent(bool reset);

        /// total data received (bytes)
        std::int64_t get_data_received(bool reset);

        /// total data (uncompressed) received (bytes)
        std::int64_t get_raw_data_received(bool reset);

        std::int64_t get_buffer_allocate_time_sent(bool reset);
        std::int64_t get_buffer_allocate_time_received(bool reset);

        std::int64_t get_pending_parcels_count(bool /*reset*/);

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        // same as above, just separated data for each action
        // number of parcels sent
        std::int64_t get_action_parcel_send_count(
            std::string const&, bool reset);

        // number of parcels received
        std::int64_t get_action_parcel_receive_count(
            std::string const&, bool reset);

        // the total time it took for all sender-side serialization operations
        // (nanoseconds)
        std::int64_t get_action_sending_serialization_time(
            std::string const&, bool reset);

        // the total time it took for all receiver-side serialization
        // operations (nanoseconds)
        std::int64_t get_action_receiving_serialization_time(
            std::string const&, bool reset);

        // total data sent (bytes)
        std::int64_t get_action_data_sent(
            std::string const&, bool reset);

        // total data received (bytes)
        std::int64_t get_action_data_received(
            std::string const&, bool reset);
#endif

        ///////////////////////////////////////////////////////////////////////
        void set_applier(applier::applier * applier)
        {
            applier_ = applier;
        }

        /// Update performance counter data
        void add_received_data(
            performance_counters::parcels::data_point const& data);

        void add_sent_data(
            performance_counters::parcels::data_point const& data);

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        void add_received_data(char const* action,
            performance_counters::parcels::data_point const& data);

        void add_sent_data(char const* action,
            performance_counters::parcels::data_point const& data);
#endif

        /// Return the configured maximal allowed message data size
        std::int64_t get_max_inbound_message_size() const
        {
            return max_inbound_message_size_;
        }

        std::int64_t get_max_outbound_message_size() const
        {
            return max_outbound_message_size_;
        }

        /// Return whether it is allowed to apply array optimizations
        bool allow_array_optimizations() const
        {
            return allow_array_optimizations_;
        }

        /// Return whether it is allowed to apply zero copy optimizations
        bool allow_zero_copy_optimizations() const
        {
            return allow_zero_copy_optimizations_;
        }

        bool enable_security() const
        {
            return enable_security_;
        }

        bool async_serialization() const
        {
            return async_serialization_;
        }

    protected:
        /// mutex for all of the member data
        mutable lcos::local::spinlock mtx_;

        hpx::applier::applier *applier_;

        /// The cache for pending parcels
#if defined(HPX_PARCELSET_PENDING_PARCELS_WORKAROUND)
        typedef util::tuple<
            std::shared_ptr<std::vector<parcel> >
          , std::vector<write_handler_type>
        >
#else
        typedef util::tuple<
            std::vector<parcel>
          , std::vector<write_handler_type>
        >
#endif
            map_second_type;
        typedef std::map<locality, map_second_type> pending_parcels_map;
        pending_parcels_map pending_parcels_;

        typedef std::set<locality> pending_parcels_destinations;
        pending_parcels_destinations parcel_destinations_;

        /// The local locality
        locality here_;

        /// The maximally allowed message size
        std::int64_t const max_inbound_message_size_;
        std::int64_t const max_outbound_message_size_;

        /// Overall parcel statistics
        performance_counters::parcels::gatherer parcels_sent_;
        performance_counters::parcels::gatherer parcels_received_;

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        // Per-action based parcel statistics
        detail::per_action_data_counter action_parcels_sent_;
        detail::per_action_data_counter action_parcels_received_;
#endif

        /// serialization is allowed to use array optimization
        bool allow_array_optimizations_;
        bool allow_zero_copy_optimizations_;

        /// enable security
        bool enable_security_;

        /// async serialization of parcels
        bool async_serialization_;

        /// priority of the parcelport
        int priority_;
        std::string type_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif
