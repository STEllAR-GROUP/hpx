//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM)
#define HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM

#include <boost/noncopyable.hpp>
#include <boost/bind.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config/forceinline.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/plugins/parcelport_factory_base.hpp>

#include <hpx/config/warnings_prefix.hpp>

#include <map>
#include <algorithm>

namespace hpx { namespace parcelset
{

    /// The \a parcelhandler is the representation of the parcelset inside a
    /// locality. It is built on top of a single parcelport. Several
    /// parcel-handlers may be connected to a single parcelport.
    class HPX_EXPORT parcelhandler : boost::noncopyable
    {
    private:
        // default callback for put_parcel
        static void default_write_handler(boost::system::error_code const&,
            parcel const& p);

        void parcel_sink(parcel const& p);

        threads::thread_state_enum decode_parcel(
            parcelport& pp, boost::shared_ptr<std::vector<char> > parcel_data,
            performance_counters::parcels::data_point receive_data);

        // make sure the parcel has been properly initialized
        void init_parcel(parcel& p)
        {
            // ensure the source locality id is set (if no component id is given)
            if (!p.source_id())
                p.set_source_id(naming::id_type(get_locality(),
                    naming::id_type::unmanaged));

            // set the current local time for this locality
            p.set_start_time(get_current_time());
        }

        typedef lcos::local::spinlock mutex_type;

    public:

        typedef std::pair<locality, std::string> handler_key_type;
        typedef std::map<
            handler_key_type, boost::shared_ptr<policies::message_handler> >
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
        parcelhandler(
            util::runtime_configuration & cfg,
            threads::threadmanager_base* tm,
            util::function_nonser<void(std::size_t, char const*)> const& on_start_thread,
            util::function_nonser<void()> const& on_stop_thread);

        ~parcelhandler() {}

        boost::shared_ptr<parcelport> get_bootstrap_parcelport() const;

        void initialize(naming::resolver_client &resolver, applier::applier *applier);

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
        /// callback functor is provided by the user.
        ///
        /// \note   The function \a put_parcel() is asynchronous.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        BOOST_FORCEINLINE void put_parcel(parcel p)
        {
            using util::placeholders::_1;
            using util::placeholders::_2;
            put_parcel(std::move(p), util::bind(
                &parcelhandler::invoke_write_handler, this, _1, _2));
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
        boost::int64_t get_parcel_send_count(std::string const&, bool) const;

        // number of messages sent
        boost::int64_t get_message_send_count(std::string const&, bool) const;

        // number of parcels routed
        boost::int64_t get_parcel_routed_count(bool);

        // number of parcels received
        boost::int64_t get_parcel_receive_count(std::string const&, bool) const;

        // number of messages received
        boost::int64_t get_message_receive_count(std::string const&, bool) const;

        // the total time it took for all sends, from async_write to the
        // completion handler (nanoseconds)
        boost::int64_t get_sending_time(std::string const&, bool) const;

        // the total time it took for all receives, from async_read to the
        // completion handler (nanoseconds)
        boost::int64_t get_receiving_time(std::string const&, bool) const;

        // the total time it took for all sender-side serialization operations
        // (nanoseconds)
        boost::int64_t get_sending_serialization_time(std::string const&, bool) const;

        // the total time it took for all receiver-side serialization
        // operations (nanoseconds)
        boost::int64_t get_receiving_serialization_time(std::string const&, bool) const;

#if defined(HPX_HAVE_SECURITY)
        // the total time it took for all sender-side security operations
        // (nanoseconds)
        boost::int64_t get_sending_security_time(std::string const&, bool) const;

        // the total time it took for all receiver-side security
        // operations (nanoseconds)
        boost::int64_t get_receiving_security_time(std::string const&, bool) const;
#endif

        // total data sent (bytes)
        boost::int64_t get_data_sent(std::string const&, bool) const;

        // total data (uncompressed) sent (bytes)
        boost::int64_t get_raw_data_sent(std::string const&, bool) const;

        // total data received (bytes)
        boost::int64_t get_data_received(std::string const&, bool) const;

        // total data (uncompressed) received (bytes)
        boost::int64_t get_raw_data_received(std::string const&, bool) const;

        boost::int64_t get_buffer_allocate_time_sent(std::string const&, bool) const;
        boost::int64_t get_buffer_allocate_time_received(std::string const&, bool) const;

        boost::int64_t get_connection_cache_statistics(std::string const& pp_type,
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
                boost::lock_guard<mutex_type> l(mtx_);
                f = write_handler_;
            }
            f(ec, p);
        }

        write_handler_type set_write_handler(write_handler_type f)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            std::swap(f, write_handler_);
            return f;
        }

    protected:
        boost::int64_t get_incoming_queue_length(bool /*reset*/) const
        {
            return 0;
        }

        boost::int64_t get_outgoing_queue_length(bool reset) const;

        std::pair<boost::shared_ptr<parcelport>, locality>
        find_appropriate_destination(naming::gid_type const & dest_gid);
        locality find_endpoint(endpoints_type const & eps, std::string const & name);

        void register_counter_types(std::string const& pp_type);

    private:
        int get_priority(std::string const& name) const
        {
            std::map<std::string, int>::const_iterator it = priority_.find(name);
            if(it == priority_.end()) return 0;
            return priority_.find(name)->second;
        }

        parcelport *find_parcelport(std::string const& type, error_code = throws) const
        {
            int priority = get_priority(type);
            if(priority <= 0) return 0;
            HPX_ASSERT(pports_.find(priority) != pports_.end());
            return pports_.find(priority)->second.get();
        }

        /// \brief Attach the given parcel port to this handler
        void attach_parcelport(boost::shared_ptr<parcelport> const& pp);

        /// The AGAS client
        naming::resolver_client *resolver_;

        /// the parcelport this handler is associated with
        typedef std::map<int, boost::shared_ptr<parcelport>,
            std::greater<int> > pports_type;
        pports_type pports_;

        std::map<std::string, int> priority_;

        /// the endpoints corresponding to the parcel-ports
        endpoints_type endpoints_;

        /// the thread-manager to use (optional)
        threads::threadmanager_base* tm_;

        /// Allow to use alternative parcel-ports (this is enabled only after
        /// the runtime systems of all localities are guaranteed to have
        /// reached a certain state).
        boost::atomic<bool> use_alternative_parcelports_;
        boost::atomic<bool> enable_parcel_handling_;

        /// Store message handlers for actions
        mutex_type handlers_mtx_;
        message_handler_map handlers_;
        bool const load_message_handlers_;

        /// Count number of (outbound) parcels routed
        boost::atomic<boost::int64_t> count_routed_;

        /// global exception handler for unhandled exceptions thrown from the
        /// parcel layer
        mutable mutex_type mtx_;
        write_handler_type write_handler_;

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


