//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM)
#define HPX_RUNTIME_RUNTIME_JUN_10_2008_1012AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/io_service_pool.hpp>
#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcelport.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/threadmanager/threadmanager.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/action_manager/action_manager.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx 
{
    /// The \a runtime class encapsulates the HPX runtime system in a simple to 
    /// use way. It makes sure all required parts of the HPX runtime system are
    /// properly initialized. 
    class runtime
    {
    public:
        /// Construct a new HPX runtime instance 
        ///
        /// \param dgas_address   [in] This is the address (IP address or 
        ///                       host name) of the locality the DGAS server is 
        ///                       running on.
        /// \param dgas_port      [in] This is the port number the DGAS server
        ///                       is listening on.
        /// \param address        [in] This is the address (IP address or 
        ///                       host name) of the locality the new runtime 
        ///                       instance should be associated with. It is 
        ///                       used for receiving parcels.
        /// \param port           [in] This is the port number the new runtime
        ///                       instance will use to listen for incoming 
        ///                       parcels.
        runtime(std::string const& dgas_address, unsigned short dgas_port,
                std::string const& address, unsigned short port) 
          : dgas_pool_(), parcel_pool_(),
            dgas_client_(dgas_pool_, dgas_address, dgas_port),
            parcel_port_(parcel_pool_, address, port),
            parcel_handler_(dgas_client_, parcel_port_),
            thread_manager_(),
            applier_(dgas_client_, parcel_handler_, thread_manager_),
            action_manager_(applier_)
        {}

        /// Construct a new HPX runtime instance 
        ///
        /// \param dgas_address   [in] This is the locality the DGAS server is 
        ///                       running on.
        /// \param address        [in] This is the locality the new runtime 
        ///                       instance should be associated with. It is 
        ///                       used for receiving parcels.
        runtime(naming::locality dgas_address, naming::locality address) 
          : dgas_pool_(), parcel_pool_(),
            dgas_client_(dgas_pool_, dgas_address),
            parcel_port_(parcel_pool_, address),
            parcel_handler_(dgas_client_, parcel_port_),
            thread_manager_(),
            applier_(dgas_client_, parcel_handler_, thread_manager_),
            action_manager_(applier_)
        {}

        /// \brief Start the runtime system
        ///
        /// \param blocking   [in] This allows to control whether this 
        ///                   call blocks until the runtime system has been 
        ///                   stopped
        ///
        /// \returns          The return value is \a true if \start() has been
        ///                   called for the first time on this runtime instance
        ///                   and will return \a false otherwise.
        bool start(bool blocking = true)
        {
            thread_manager_.run();
            return parcel_port_.run(blocking);     // starts parcel_pool_ as well
        }

        /// Stop the runtime system
        ///
        /// \param blocking   [in] This allows to control whether this 
        ///                   call blocks until the runtime system has been 
        ///                   fully stopped. If this parameter is \a false then 
        ///                   this call will initialize the stop action but will
        ///                   return immediately. Use a second call to stop 
        ///                   with this parameter set to \a true to wait for 
        ///                   all internal work to be completed.
        void stop(bool blocking = true)
        {
            parcel_port_.stop(blocking);    // stops parcel_pool_ as well
            thread_manager_.stop(blocking);
            dgas_pool_.stop();
        }

        ///////////////////////////////////////////////////////////////////////

        /// 
        naming::resolver_client const& get_dgas_client() const
        {
            return dgas_client_;
        }

        /// 
        parcelset::parcelhandler& get_parcel_handler()
        {
            return parcel_handler_;
        }

        /// 
        threadmanager::threadmanager& get_thread_manager()
        {
            return thread_manager_;
        }

        /// 
        applier::applier& get_applier()
        {
            return applier_;
        }

        /// 
        action_manager::action_manager& get_action_manager()
        {
            return action_manager_;
        }

        /// \brief Allow access to the locality this runtime instance is 
        /// associated with.
        ///
        /// This accessor returns a reference to the locality this runtime
        /// instance is associated with.
        naming::locality const& here() const
        {
            return parcel_port_.here();
        }

    private:
        util::io_service_pool dgas_pool_; 
        util::io_service_pool parcel_pool_; 
        naming::resolver_client dgas_client_;
        parcelset::parcelport parcel_port_;
        parcelset::parcelhandler parcel_handler_;
        threadmanager::threadmanager thread_manager_;
        applier::applier applier_;
        action_manager::action_manager action_manager_;
    };
}

#endif
