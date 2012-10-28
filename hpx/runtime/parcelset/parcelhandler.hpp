//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM)
#define HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM

#include <boost/noncopyable.hpp>
#include <boost/bind.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/locality.hpp>

#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelhandler_queue_base.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/logging.hpp>

#include <hpx/config/warnings_prefix.hpp>

#include <map>

namespace hpx { namespace parcelset
{
    /// The \a parcelhandler is the representation of the parcelset inside a
    /// locality. It is built on top of a single parcelport. Several
    /// parcel-handlers may be connected to a single parcelport.
    class HPX_EXPORT parcelhandler : boost::noncopyable
    {
    private:
        static void default_write_handler(boost::system::error_code const&,
            std::size_t /*size*/) {}

        void parcel_sink(parcelport& pp,
            boost::shared_ptr<std::vector<char> > parcel_data,
            threads::thread_priority priority,
            performance_counters::parcels::data_point const& receive_data);

        threads::thread_state_enum decode_parcel(
            parcelport& pp, boost::shared_ptr<std::vector<char> > parcel_data,
            performance_counters::parcels::data_point receive_data);

        // make sure the parcel has been properly initialized
        void init_parcel(parcel& p)
        {
            // ensure the source locality id is set (if no component id is given)
            if (!p.get_source())
                p.set_source(naming::id_type(locality_, naming::id_type::unmanaged));

            // set the current local time for this locality
            p.set_start_time(get_current_time());
        }

        // find and return the specified parcelport
        parcelport* find_parcelport(connection_type type,
            error_code& ec = throws) const;

    public:
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
        parcelhandler(naming::resolver_client& resolver, parcelport& pp,
            threads::threadmanager_base* tm, parcelhandler_queue_base* policy);

        ~parcelhandler()
        {
        }

        /// \brief Attach the given parcel port to this handler
        void attach_parcelport(parcelport& pp);

        /// \brief Allow access to AGAS resolver instance.
        ///
        /// This accessor returns a reference to the AGAS resolver client
        /// object the parcelhandler has been initialized with (see
        /// parcelhandler constructors). This is the same resolver instance
        /// this parcelhandler has been initialized with.
        naming::resolver_client& get_resolver();

        /// Allow access to parcelport instance.
        ///
        /// This accessor returns a reference to the parcelport object
        /// the parcelhandler has been initialized with (see parcelhandler
        /// constructors). This is the same \a parcelport instance this
        /// parcelhandler has been initialized with.
        parcelport& get_parcelport()
        {
            return *find_parcelport(connection_tcpip);
        }

        /// Return the locality_id of this locality
        ///
        /// This accessor allows to retrieve the locality_id value being assigned to
        /// the locality this parcelhandler is associated with. This returns the
        /// same value as would be returned by:
        ///
        /// \code
        ///     naming::id_type locality_id;
        ///     get_resolver().get_locality_id(here, locality_id);
        /// \endcode
        ///
        /// but doesn't require the full AGAS round trip as the prefix value
        /// is cached inside the parcelhandler.
        naming::gid_type const& get_locality() const
        {
            return locality_;
        }

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
            components::component_type type = components::component_invalid) const;

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
            components::component_type type = components::component_invalid) const;

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
        void sync_put_parcel(parcel& p);

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
        void put_parcel(parcel& p, write_handler_type f);

        /// This put_parcel() function overload is asynchronous, but no
        /// callback functor is provided by the user.
        ///
        /// \note   The function \a put_parcel() is asynchronous.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The
        ///                 parcel \a p will be modified in place, as it will
        ///                 get set the resolved destination address and parcel
        ///                 id (if not already set).
        void put_parcel(parcel& p)
        { put_parcel(p, &parcelhandler::default_write_handler); }

        /// The function \a get_parcel returns the next available parcel
        ///
        /// \param p        [out] The parcel instance to be filled with the
        ///                 received parcel. If the functioned returns \a true
        ///                 this will be the next received parcel.
        ///
        /// \returns        Returns \a true if the next parcel has been
        ///                 retrieved successfully. The reference given by
        ///                 parameter \a p will be initialized with the
        ///                 received parcel data.
        ///                 Return \a false if no parcel is available in the
        ///                 parcelhandler, the reference \a p is not touched.
        ///
        /// The returned parcel will be no longer available from the
        /// parcelhandler as it is removed from the internal queue of received
        /// parcels.
        bool get_parcel(parcel& p)
        {
            return parcels_->get_parcel(p);
        }

        /// The function \a get_parcel returns the next available parcel
        ///
        /// \param p        [out] The parcel instance to be filled with the
        ///                 received parcel. If the functioned returns \a true
        ///                 this will be the next received parcel.
        /// \param parcel_id  [in] The id of the parcel to fetch
        ///
        /// \returns        Returns \a true if the parcel with the given id
        ///                 has been retrieved successfully. The reference
        ///                 given by parameter \a p will be initialized with
        ///                 the received parcel data.
        ///                 Return \a false if no parcel is available in the
        ///                 parcelhandler, the reference \a p is not touched.
        ///
        /// The returned parcel will be no longer available from the
        /// parcelhandler as it is removed from the internal queue of received
        /// parcels.
        bool get_parcel(parcel& p, naming::gid_type const& parcel_id)
        {
            return parcels_->get_parcel(p, parcel_id);
        }

        /// Register an event handler to be called whenever a parcel has been
        /// received
        ///
        /// \param sink     [in] A function object to be invoked whenever a
        ///                 parcel has been received by the parcelhandler. It is
        ///                 possible to register more than one (different)
        ///                 function object. The signature of this function
        ///                 object is expected to be:
        ///
        /// \code
        ///      void sink (hpx::parcelset::parcelhandler& pp
        ///                 hpx::naming::address const&);
        /// \endcode
        ///
        ///                 where \a pp is a reference to the parcelhandler this
        ///                 function object instance is invoked by, and \a dest
        ///                 is the local destination address of the parcel.
        bool register_event_handler(
            parcelhandler_queue_base::callback_type const& sink)
        {
            return parcels_->register_event_handler(sink);
        }

        /// Register an event handler to be called whenever a parcel has been
        /// received
        ///
        /// \param sink     [in] A function object to be invoked whenever a
        ///                 parcel has been received by the parcelhandler. It is
        ///                 possible to register more than one (different)
        ///                 function object. The signature of this function
        ///                 object is expected to be:
        ///
        /// \code
        ///      void sink (hpx::parcelset::parcelhandler& pp
        ///                 hpx::naming::address const&);
        /// \endcode
        ///
        ///                 where \a pp is a reference to the parcelhandler this
        ///                 function object instance is invoked by, and \a dest
        ///                 is the local destination address of the parcel.
        /// \param conn     [in] A instance of a unspecified type allowing to
        ///                 manage the lifetime of the established connection.
        ///                 The easiest way is to pass an instance of \a
        ///                 scoped_connection_type allowing to automatically
        ///                 unregister this connection whenever the connection
        ///                 instance goes out of scope.
        bool register_event_handler(
            parcelhandler_queue_base::callback_type const& sink
          , parcelhandler_queue_base::connection_type& conn)
        {
            return parcels_->register_event_handler(sink, conn);
        }

        /// The 'scoped_connection_type' typedef simplifies to manage registered
        /// event handlers. Instances of this type may be passed as the second
        /// parameter to the \a register_event_handler() function
        typedef parcelhandler_queue_base::connection_type scoped_connection_type;

        double get_current_time() const
        {
            return util::high_resolution_timer::now();
        }

        /// \brief Allow access to the locality of the parcelport this
        /// parcelhandler is associated with.
        ///
        /// This accessor returns a reference to the locality of the parcelport
        /// this parcelhandler is associated with.
        naming::locality const& here() const
        {
            return find_parcelport(connection_tcpip)->here();
        }

        /// The function register_counter_types() is called during startup to
        /// allow the registration of all performance counter types for this
        /// parcel-handler instance.
        void register_counter_types(connection_type pp_type = connection_tcpip);

    protected:
        std::size_t get_incoming_queue_length() const
        {
            return parcels_->get_queue_length();
        }

        std::size_t get_outgoing_queue_length() const
        {
            return find_parcelport(connection_tcpip)->get_pending_parcels_count();
        }

    private:
        /// The AGAS client
        naming::resolver_client& resolver_;

        /// The site prefix of the locality
        naming::gid_type locality_;

        /// the parcelport this handler is associated with
        std::map<connection_type, parcelport*> pports_;

        /// the thread-manager to use (optional)
        threads::threadmanager_base* tm_;

        ///
        boost::shared_ptr<parcelhandler_queue_base> parcels_;
    };

}}

#include <hpx/config/warnings_suffix.hpp>

#endif


