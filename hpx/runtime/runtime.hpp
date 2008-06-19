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
    /// This is the signature expected to be exposed by a function registered 
    /// as HPX's 'main' function. This main function is the easiest way of 
    /// bootstrapping an application
    typedef threadmanager::thread_state hpx_main_function_type(
        threadmanager::px_thread_self&, applier::applier&);

    namespace detail
    {
        /// The mainfunc helper template is used to encapsulate the 
        /// function or function object passed to runtime#start to avoid using
        /// boost::bind for this purpose (boost::bind doesn't work well if it
        /// gets another function object passed, which has been constructed by
        /// boost::bind as well).
        struct mainfunc
        {
            mainfunc(boost::function<hpx_main_function_type> func, 
                    applier::applier& appl)
              : func_(func), appl_(appl)
            {}

            threadmanager::thread_state 
            operator()(threadmanager::px_thread_self& self)
            {
                return func_(self, appl_);
            }
            
            boost::function<hpx_main_function_type> func_;
            applier::applier& appl_;
        };
    }

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

        ~runtime()
        {
            // stop all services
            parcel_port_.stop();
            thread_manager_.stop();
            dgas_pool_.stop();
        }

        /// \brief Start the runtime system
        ///
        /// \param func       [in] This is the main function of an HPX 
        ///                   application. It will be scheduled for execution
        ///                   by the thread manager as soon as the runtime has 
        ///                   been initialized. This function is expected to 
        ///                   expose an interface as defined by the typedef
        ///                   \a hpx_main_function_type.
        /// \param blocking   [in] This allows to control whether this 
        ///                   call blocks until the runtime system has been 
        ///                   stopped
        void start(boost::function<hpx_main_function_type> func, 
            bool blocking = false)
        {
            // start services (service threads)
            thread_manager_.run();        // start the thread manager
            parcel_port_.run(false);      // starts parcel_pool_ as well

            // register the given main function with the thread manager
            thread_manager_.register_work(detail::mainfunc(func, applier_));

            // block if required
            if (blocking)
                parcel_port_.run(true);
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

}   // namespace hpx

#endif
