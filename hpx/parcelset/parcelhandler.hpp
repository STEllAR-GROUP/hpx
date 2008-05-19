//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM)
#define HPX_PARCELSET_PARCELHANDLER_MAY_18_2008_0935AM

#include <boost/noncopyable.hpp>
#include <boost/signals.hpp>
#include <boost/bind.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/naming/name.hpp>
#include <hpx/naming/address.hpp>
#include <hpx/naming/locality.hpp>
#include <hpx/naming/resolver_client.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/parcelset/parcelport.hpp>
#include <hpx/parcelset/server/parcelhandler_queue.hpp>

namespace hpx { namespace parcelset
{
    /// The parcelhandler is the representation of the parcelset inside a 
    /// locality. It is built on top of a single parcelport. Several 
    /// parcelhandler's may be connected to a parcelport
    class parcelhandler : boost::noncopyable
    {
    private:
        static void default_write_handler(boost::system::error_code const&, 
            std::size_t) {}

        void parcel_sink(parcelport& pp, parcel const& p)
        {
            parcels_.add_parcel(p);
        }
        
        // avoid warnings about using \a this in member initializer list
        parcelhandler& This() { return *this; }
        
    public:
        /// \param resolver [in] A reference to the DGAS client to use for 
        ///                 address translation requests to be made by the 
        ///                 parcelhandler.
        parcelhandler(naming::resolver_client& resolver, parcelport& pp) 
          : resolver_(resolver), pp_(pp), parcels_(This()),
            id_range_(pp.here(), resolver),
            startup_time_(util::high_resolution_timer::now()), timer_()
        {
            // retrieve the prefix to be used for this site
            resolver_.get_prefix(pp.here(), prefix_);    // throws on error

            // register our callback function with the parcelport
            pp_.register_event_handler(
                boost::bind(&parcelhandler::parcel_sink, this, _1, _2), conn_);
        }
        ~parcelhandler() 
        {
        }
        
        /// Allow access to DGAS resolver instance. 
        ///
        /// This accessor returns a reference to the DGAS resolver client object
        /// the parcelhandler has been initialized with (see parcelhandler 
        /// constructors).
        naming::resolver_client& get_resolver()
        {
            return resolver_;
        }
        
        /// Allow access to parcelport instance. 
        ///
        /// This accessor returns a reference to the parcelport object
        /// the parcelhandler has been initialized with (see parcelhandler 
        /// constructors).
        parcelport& get_parcelport()
        {
            return pp_;
        }
        
        /// Return the prefix of this locality
        ///
        /// This accessor allows to retrieve the prefix value being assigned to 
        /// the locality this parcelhandler is associated with. This returns the
        /// same value as would be returned by:
        ///
        /// \code
        ///     naming::id_type prefix;
        ///     get_resolver().get_prefix(here, prefix);
        /// \endcode
        /// 
        /// but doesn't require the fully DGAS round trip as the prefix value 
        /// is cached inside the parcelhandler.
        naming::id_type const& get_prefix() const 
        { 
            return prefix_; 
        }

        /// A parcel is submitted for transport at the source locality site to 
        /// the parcel set of the locality with the put-parcel command
        ///
        /// \note The function sync_put_parcel() is synchronous.
        ///
        /// \param p        [in, out] A reference to the parcel to send. The 
        ///                 function does not return before the parcel has been
        ///                 transmitted. The parcel \a p will be modified in 
        ///                 place, as it will get set the resolved destination
        ///                 address and parcel id (if not already set).
        parcel_id sync_put_parcel(parcel& p);
        
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
        /// \code
        ///     void f (boost::system::error_code const& err, std::size_t );
        /// \endcode
        ///
        ///                 where \a err is the status code of the operation and
        ///                       \a size is the number of successfully 
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
            
            // set the current local time for this locality
            p.set_start_time(startup_time_ + timer_.elapsed());
            
            // send the parcel to its destination
            pp_.put_parcel(p, f);
            
            // return parcel id of the parcel being sent
            return p.get_parcel_id();
        }

        /// This put_parcel() function overload is asynchronous, but no 
        /// callback functor is provided by the user. 
        ///
        /// \param p        [in, out] A reference to the parcel to send. The 
        ///                 parcel \a p will be modified in place, as it will 
        ///                 get set the resolved destination address and parcel 
        ///                 id (if not already set).
        parcel_id put_parcel(parcel& p)
        {
            return put_parcel(p, &parcelhandler::default_write_handler);
        }

        /// The get_parcel command returns a parcel, or if the parcel set is 
        /// empty then false is returned. 
        ///
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the next parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no parcel is available in the parcelhandler
        ///
        /// The returned parcel will be no longer available from the parcelhandler
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(parcel& p)
        {
            return parcels_.get_parcel(p);
        }

        /// Parcels may be typed by the action class they wish to invoke. Return
        /// next parcel of given action class (FIXME: shouldn't we add the 
        /// component type here as well?)
        
        /// This get_parcel() overload returns the next available parcel 
        /// addressed to any instance of a component of type \a c.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelhandler.
        ///
        /// \param c        [in] The component type the parcel has to be 
        ///                 addressed to.
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the next parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no corresponding parcel is available in 
        ///                 the parcelhandler
        ///
        /// The returned parcel will be no longer available from the parcelhandler
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(components::component_type c, parcel& p)
        {
            return parcels_.get_parcel(c, p);
        }
        
        /// This get_parcel() overload returns the parcel with the given parcel 
        /// tag (id).
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelhandler.
        ///
        /// \param tag      [in] The parcel tag (id) of the parcel to retrieve.
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no corresponding parcel is available in 
        ///                 the parcelhandler
        ///
        /// The returned parcel will be no longer available from the parcelhandler
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel(parcel_id tag, parcel& p)
        {
            return parcels_.get_parcel(tag, p);
        }
        
        /// This get_parcel() overload returns the parcel being sent from the 
        /// locality with the given source id.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelhandler.
        ///
        /// \param source   [in] The id of the source locality.
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no corresponding parcel is available in 
        ///                 the parcelhandler
        ///
        /// The returned parcel will be no longer available from the parcelhandler
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel_from(naming::id_type source, parcel& p)
        {
            return parcels_.get_parcel_from(source, p);
        }
        
        /// This get_parcel() overload returns the parcel being to the given
        /// destination address.
        ///
        /// The function get_pacel() is synchronous, i.e. it will return only
        /// after the parcel has been retrieved from the parcelhandler.
        ///
        /// \param dest     [in] The id of the destination component.
        /// \param p        [out] The parcel instance to be filled with the 
        ///                 received parcel. If the functioned returns \a true 
        ///                 this will be the next received parcel.
        ///
        /// \returns        \a true if the parcel has been retrieved 
        ///                 successfully. 
        ///                 \a false if no corresponding parcel is available in 
        ///                 the parcelhandler
        ///
        /// The returned parcel will be no longer available from the parcelhandler
        /// as it is removed from the internal queue of received parcels.
        bool get_parcel_for(naming::id_type dest, parcel& p)
        {
            return parcels_.get_parcel_for(dest, p);
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
        ///                 hpx::parcelset::parcel const& p);
        /// \endcode
        ///
        ///                 where \a pp is a reference to the parcelhandler this
        ///                 function object instance is invoked by, and \a dest
        ///                 is the local destination address of the parcel.
        template <typename F>
        bool register_event_handler(F sink)
        {
            return parcels_.register_event_handler(sink);
        }

        template <typename F, typename Connection>
        bool register_event_handler(F sink, Connection& conn)
        {
            return parcels_.register_event_handler(sink, conn);
        }

    protected:        
        // generate next unique id
        parcel_id get_next_id()
        {
            return id_range_.get_id();
        }

    private:
        /// The DGAS client
        naming::resolver_client& resolver_;
        
        /// The site prefix of the locality 
        naming::id_type prefix_;

        /// the parcelport this handler is associated with
        parcelport& pp_;

        /// 
        server::parcelhandler_queue parcels_;
        
        /// The site current range of ids to be used for id_type instances
        util::unique_ids id_range_;
        
        boost::signals::scoped_connection conn_;

        /// This is the timer instance for this parcelhandler
        double startup_time_;
        util::high_resolution_timer timer_;
    };

}}

#endif


