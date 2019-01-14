//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM)
#define HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind_front.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util_fwd.hpp>

#include <hpx/plugins/parcelport_factory_base.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace parcelset
{
    // default callback for put_parcel
    void default_write_handler(boost::system::error_code const&,
        parcel const& p);

    /// The \a parcelhandler is the representation of the parcelset inside a
    /// locality. It is built on top of a single parcelport. Several
    /// parcel-handlers may be connected to a single parcelport.
    class HPX_EXPORT parcelhandler
    {
    public:
        HPX_NON_COPYABLE(parcelhandler);

    private:
        void parcel_sink(parcel const& p);

        threads::thread_state_enum decode_parcel(
            parcelport& pp, std::shared_ptr<std::vector<char> > parcel_data,
            performance_counters::parcels::data_point receive_data);

        // make sure the parcel has been properly initialized
        void init_parcel(parcel& p);

        typedef lcos::local::spinlock mutex_type;

    public:

        typedef std::pair<locality, std::string> handler_key_type;
        typedef std::map<
            handler_key_type, std::shared_ptr<policies::message_handler> >
        message_handler_map;

        typedef parcelport::read_handler_type read_handler_type;
        typedef parcelport::write_handler_type write_handler_type;

        /// Construct a new \a parcelhandler initializing it from a AGAS client
        /// instance (parameter \a resolver) and the parcelport to be used for
        /// parcel send and receive (parameter \a pp).
        ///
        /// \param resolver [in] A reference to the AGAS client to use for
        ///                 address translation requests to be made by the
        ///                 parcelhandler.
        /// \param pp       [in] A reference to the \a parcelport this \a
        ///                 parcelhandler is connected to. This \a parcelport
        ///                 instance will be used for any parcel related
        ///                 transport operations the parcelhandler carries out.
        parcelhandler(util::runtime_configuration& cfg,
            threads::threadmanager* tm,
            util::function_nonser<void(std::size_t, char const*)> const&
                on_start_thread,
            util::function_nonser<void(std::size_t, char const*)> const&
                on_stop_thread);

        ~parcelhandler() {}

        std::shared_ptr<parcelport> get_bootstrap_parcelport() const;

        void initialize(naming::resolver_client &resolver, applier::applier *applier);

        void flush_parcels();

        /// \brief Stop all parcel ports associated with this parcelhandler
        void stop(bool blocking = true);

        /// \brief do background work in the parcel layer
        ///
        /// \returns Whether any work has been performed
        bool do_background_work(std::size_t num_thread = 0,
            bool stop_buffering = false);

        /// \brief Allow access to AGAS resolver instance.
        ///
        /// This accessor returns a reference to the AGAS resolver client
        /// object the parcelhandler has been initialized with (see
        /// parcelhandler constructors). This is the same resolver instance
        /// this parcelhandler has been initialized with.
        naming::resolver_client& get_resolver();

        /// Return the list of all remote localities supporting the given
        /// component type
        ///
        /// \param prefixes [out] The reference to a vector of id_types filled
        ///                 by the function.
        /// \param type     [in] The type of the component which needs to exist
        ///                 on the returned localities.
        ///
        /// \returns The function returns \a true if there is at least one
        ///          remote locality known by AGAS
        ///          (!prefixes.empty()).
        bool get_raw_remote_localities(std::vector<naming::gid_type>& locality_ids,
            components::component_type type = components::component_invalid,
            error_code& ec = throws) const;

        /// Return the list of all localities supporting the given
        /// component type
        ///
        /// \param prefixes [out] The reference to a vector of id_types filled
        ///                 by the function.
        /// \param type     [in] The type of the component which needs to exist
        ///                 on the returned localities.
        ///
        /// \returns The function returns \a true if there is at least one
        ///          locality known by AGAS
        ///          (!prefixes.empty()).
        bool get_raw_localities(std::vector<naming::gid_type>& locality_ids,
            components::component_type type, error_code& ec = throws) const;

        /// A parcel is submitted for transport at the source locality site to
        /// the parcel set of the locality with the put-parcel command
        ///
        /// \note The function \a sync_put_parcel() is synchronous, it blocks
        ///       until the parcel has been sent by the underlying \a
        ///       parcelport.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 function does not return before the parcel has been
        ///                 transmitted. The parcel \a p will be modified in
        ///                 place, as it will get set the resolved destination
        ///                 address and parcel id (if not already set).
        void sync_put_parcel(parcel p);

        /// A parcel is submitted for transport at the source locality site to
        /// the parcel set of the locality with the put-parcel command
        //
        /// \note The function \a put_parcel() is asynchronous, the provided
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
        ///     void f (boost::system::error_code const& err, std::size_t );
        /// \endcode
        ///
        ///                 where \a err is the status code of the operation and
        ///                       \a size is the number of successfully
        ///                              transferred bytes.
        void put_parcel(parcel p, write_handler_type f);

        /// This put_parcel() function overload is asynchronous, but no
        /// callback is provided by the user.
        ///
        /// \note   The function \a put_parcel() is asynchronous.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        HPX_FORCEINLINE void put_parcel(parcel p)
        {
            put_parcel(std::move(p), util::bind_front(
                &parcelhandler::invoke_write_handler, this));
        }

        /// A parcel is submitted for transport at the source locality site to
        /// the parcel set of the locality with the put-parcel command
        //
        /// \note The function \a put_parcel() is asynchronous, the provided
        /// function or function object gets invoked on completion of the send
        /// operation or on any error.
        ///
        /// \param p        [in] The parcels to send.
        /// \param f        [in] The function objects to be invoked on
        ///                 successful completion or on errors. The signature
        ///                 of these function object are expected to be:
        ///
        /// \code
        ///     void f (boost::system::error_code const& err, std::size_t );
        /// \endcode
        ///
        ///                 where \a err is the status code of the operation and
        ///                       \a size is the number of successfully
        ///                              transferred bytes.
        void put_parcels(std::vector<parcel> p, std::vector<write_handler_type> f);

        /// This put_parcel() function overload is asynchronous, but no
        /// callback is provided by the user.
        ///
        /// \note   The function \a put_parcel() is asynchronous.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        void put_parcels(std::vector<parcel> parcels)
        {
            std::vector<write_handler_type> handlers(parcels.size(),
                util::bind_front(&parcelhandler::invoke_write_handler, this));

            put_parcels(std::move(parcels), std::move(handlers));
        }

        double get_current_time() const
        {
            return util::high_resolution_timer::now();
        }

        /// \brief Factory function used in serialization to create a given
        /// locality endpoint
        locality create_locality(std::string const & name) const
        {
            return find_parcelport(name)->create_locality();
        }

        /// Return the name of this locality as retrieved from the active
        /// parcelport
        std::string get_locality_name() const;

        ///////////////////////////////////////////////////////////////////////
        /// The function register_counter_types() is called during startup to
        /// allow the registration of all performance counter types for this
        /// parcel-handler instance.
        void register_counter_types();

        /// \brief Make sure the specified locality is not held by any
        /// connection caches anymore
        void remove_from_connection_cache(naming::gid_type const& gid,
            endpoints_type const& endpoints);

        /// \brief return the endpoints associated with this parcelhandler
        /// \returns all connection information for the enabled parcelports
        endpoints_type const & endpoints() const
        {
            return endpoints_;
        }

        void enable_alternative_parcelports()
        {
            use_alternative_parcelports_.store(true);
        }

        void disable_alternative_parcelports()
        {
            use_alternative_parcelports_.store(false);
        }

        /// Return the reference to an existing io_service
        util::io_service_pool* get_thread_pool(char const* name);

        ///////////////////////////////////////////////////////////////////////
        policies::message_handler* get_message_handler(char const* action,
            char const* message_handler_type, std::size_t num_messages,
             std::size_t interval, locality const& loc,
             error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        // Performance counter data

        // number of parcels sent
        std::int64_t get_parcel_send_count(
            std::string const& pp_type, bool reset) const;

        // number of messages sent
        std::int64_t get_message_send_count(
            std::string const& pp_type, bool reset) const;

        // number of parcels routed
        std::int64_t get_parcel_routed_count(bool reset);

        // number of parcels received
        std::int64_t get_parcel_receive_count(
            std::string const& pp_type, bool reset) const;

        // number of messages received
        std::int64_t get_message_receive_count(
            std::string const& pp_type, bool reset) const;

        // the total time it took for all sends, from async_write to the
        // completion handler (nanoseconds)
        std::int64_t get_sending_time(
            std::string const& pp_type, bool reset) const;

        // the total time it took for all receives, from async_read to the
        // completion handler (nanoseconds)
        std::int64_t get_receiving_time(
            std::string const& pp_type, bool reset) const;

        // the total time it took for all sender-side serialization operations
        // (nanoseconds)
        std::int64_t get_sending_serialization_time(
            std::string const& pp_type, bool reset) const;

        // the total time it took for all receiver-side serialization
        // operations (nanoseconds)
        std::int64_t get_receiving_serialization_time(
            std::string const& pp_type, bool reset) const;

        // total data sent (bytes)
        std::int64_t get_data_sent(
            std::string const& pp_type, bool reset) const;

        // total data (uncompressed) sent (bytes)
        std::int64_t get_raw_data_sent(
            std::string const& pp_type, bool reset) const;

        // total data received (bytes)
        std::int64_t get_data_received(
            std::string const& pp_type, bool reset) const;

        // total data (uncompressed) received (bytes)
        std::int64_t get_raw_data_received(
            std::string const& pp_type, bool reset) const;

        std::int64_t get_buffer_allocate_time_sent(
            std::string const& pp_type, bool reset) const;

        std::int64_t get_buffer_allocate_time_received(
            std::string const& pp_type, bool reset) const;

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        // same as above, just separated data for each action
        // number of parcels sent
        std::int64_t get_action_parcel_send_count(
            std::string const& pp_type, std::string const& action,
            bool reset) const;

        // number of parcels received
        std::int64_t get_action_parcel_receive_count(
            std::string const& pp_type, std::string const& action,
            bool reset) const;

        // the total time it took for all sender-side serialization operations
        // (nanoseconds)
        std::int64_t get_action_sending_serialization_time(
            std::string const& pp_type, std::string const& action,
            bool reset) const;

        // the total time it took for all receiver-side serialization
        // operations (nanoseconds)
        std::int64_t get_action_receiving_serialization_time(
            std::string const& pp_type, std::string const& action,
            bool reset) const;

        // total data sent (bytes)
        std::int64_t get_action_data_sent(std::string const& pp_type,
            std::string const& action, bool reset) const;

        // total data received (bytes)
        std::int64_t get_action_data_received(std::string const& pp_type,
            std::string const& action, bool reset) const;
#endif

        //
        std::int64_t get_connection_cache_statistics(std::string const& pp_type,
            parcelport::connection_cache_statistics_type stat_type, bool) const;

        void list_parcelports(std::ostringstream& strm) const;
        void list_parcelport(std::ostringstream& strm,
            std::string const& ppname, int priority, bool bootstrap) const;

        // manage default exception handler
        void invoke_write_handler(
            boost::system::error_code const& ec, parcel const & p) const
        {
            write_handler_type f;
            {
                std::lock_guard<mutex_type> l(mtx_);
                f = write_handler_;
            }
            f(ec, p);
        }

        write_handler_type set_write_handler(write_handler_type f)
        {
            std::lock_guard<mutex_type> l(mtx_);
            std::swap(f, write_handler_);
            return f;
        }

    protected:
        std::int64_t get_incoming_queue_length(bool /*reset*/) const
        {
            return 0;
        }

        std::int64_t get_outgoing_queue_length(bool reset) const;

        std::pair<std::shared_ptr<parcelport>, locality>
        find_appropriate_destination(naming::gid_type const & dest_gid);
        locality find_endpoint(endpoints_type const & eps, std::string const & name);

        void register_counter_types(std::string const& pp_type);
        void register_connection_cache_counter_types(std::string const& pp_type);

    private:
        int get_priority(std::string const& name) const
        {
            std::map<std::string, int>::const_iterator it = priority_.find(name);
            if(it == priority_.end()) return 0;
            return priority_.find(name)->second;
        }

        parcelport *find_parcelport(std::string const& type, error_code& = throws) const
        {
            int priority = get_priority(type);
            if(priority <= 0) return nullptr;
            HPX_ASSERT(pports_.find(priority) != pports_.end());
            return pports_.find(priority)->second.get();
        }

        /// \brief Attach the given parcel port to this handler
        void attach_parcelport(std::shared_ptr<parcelport> const& pp);

        /// The AGAS client
        naming::resolver_client *resolver_;

        /// the parcelport this handler is associated with
        typedef std::map<int, std::shared_ptr<parcelport>,
            std::greater<int> > pports_type;
        pports_type pports_;

        std::map<std::string, int> priority_;

        /// the endpoints corresponding to the parcel-ports
        endpoints_type endpoints_;

        /// the thread-manager to use (optional)
        threads::threadmanager* tm_;

        /// Allow to use alternative parcel-ports (this is enabled only after
        /// the runtime systems of all localities are guaranteed to have
        /// reached a certain state).
        std::atomic<bool> use_alternative_parcelports_;
        std::atomic<bool> enable_parcel_handling_;

        /// Store message handlers for actions
        mutex_type handlers_mtx_;
        message_handler_map handlers_;
        bool const load_message_handlers_;

        /// Count number of (outbound) parcels routed
        std::atomic<std::int64_t> count_routed_;

        /// global exception handler for unhandled exceptions thrown from the
        /// parcel layer
        mutable mutex_type mtx_;
        write_handler_type write_handler_;

        /// cache whether networking has been enabled
        bool is_networking_enabled_;

    private:
        static std::vector<plugins::parcelport_factory_base *> &
            get_parcelport_factories();

    public:
        static void add_parcelport_factory(plugins::parcelport_factory_base *);

        static void init(int *argc, char ***argv, util::command_line_handling &cfg);

        /// load runtime configuration settings ...
        static std::vector<std::string> load_runtime_configuration();
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif


