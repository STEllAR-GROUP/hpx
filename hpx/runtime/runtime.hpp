//  Copyright (c) 2007-2009 Hartmut Kaiser
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
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/action_manager.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx 
{
    /// \class runtime runtime.hpp hpx/runtime/runtime.hpp
    ///
    /// The \a runtime class encapsulates the HPX runtime system in a simple to 
    /// use way. It makes sure all required parts of the HPX runtime system are
    /// properly initialized. 
    class HPX_EXPORT runtime
    {
    private:
        // avoid warnings about usage of this in member initializer list
        runtime* This() { return this; }

        // init the TSS for the applier
        void init_applier();

    public:
        /// The \a hpx_main_function_type is the default function type usable 
        /// as the main HPX thread function.
        typedef int hpx_main_function_type();

        /// A HPX runtime can be executed in two different modes: console mode
        /// and worker mode.
        enum mode
        {
            console = 0,    ///< The runtime instance represents the application console
            worker = 1      ///< The runtime instance represents a worker locality
        };

        /// Construct a new HPX runtime instance 
        ///
        /// \param address        [in] This is the address (IP address or 
        ///                       host name) of the locality the new runtime 
        ///                       instance should be associated with. It is 
        ///                       used for receiving parcels.
        /// \param port           [in] This is the port number the new runtime
        ///                       instance will use to listen for incoming 
        ///                       parcels.
        /// \param agas_address   [in] This is the address (IP address or 
        ///                       host name) of the locality the AGAS server is 
        ///                       running on. If this value is not 
        ///                       specified the actual address will be 
        ///                       taken from the configuration file (hpx.ini).
        /// \param agas_port      [in] This is the port number the AGAS server 
        ///                       is listening on. If this value is not 
        ///                       specified the actual port number will be 
        ///                       taken from the configuration file (hpx.ini).
        /// \param locality_mode  [in] This is the mode the given runtime 
        ///                       instance should be executed in.
        runtime(std::string const& address, boost::uint16_t port,
                std::string const& agas_address = "", 
                boost::uint16_t agas_port = 0, mode  locality_mode = console);

        /// Construct a new HPX runtime instance 
        ///
        /// \param address        [in] This is the locality the new runtime 
        ///                       instance should be associated with. It is 
        ///                       used for receiving parcels. 
        /// \note The AGAS locality to use will be taken from the configuration 
        ///       file (hpx.ini).
        runtime(naming::locality address, mode locality_mode = worker);

        /// Construct a new HPX runtime instance 
        ///
        /// \param address        [in] This is the locality the new runtime 
        ///                       instance should be associated with. It is 
        ///                       used for receiving parcels. 
        /// \param agas_address   [in] This is the locality the AGAS server is 
        ///                       running on. 
        runtime(naming::locality address, naming::locality agas_address, 
            mode locality_mode = worker);

        /// \brief The destructor makes sure all HPX runtime services are 
        ///        properly shut down before existing.
        ~runtime();

        /// \brief Start the runtime system
        ///
        /// \param func       [in] This is the main function of an HPX 
        ///                   application. It will be scheduled for execution
        ///                   by the thread manager as soon as the runtime has 
        ///                   been initialized. This function is expected to 
        ///                   expose an interface as defined by the typedef
        ///                   \a hpx_main_function_type.
        /// \param num_threads [in] The initial number of threads to be started 
        ///                   by the threadmanager. This parameter is optional 
        ///                   and defaults to 1.
        /// \param blocking   [in] This allows to control whether this 
        ///                   call blocks until the runtime system has been 
        ///                   stopped. If this parameter is \a true the 
        ///                   function \a runtime#start internally will call 
        ///                   \a runtime#wait.
        ///
        /// \returns          This function will return the value as returned 
        ///                   as the result of the invocation of the function 
        ///                   object given by the parameter \p func.
        int start(boost::function<hpx_main_function_type> func, 
            std::size_t num_threads = 1, bool blocking = false);

        /// \brief Wait for the shutdown action to be executed
        void wait();

        /// \brief Stop the runtime system
        ///
        /// \param blocking   [in] This allows to control whether this 
        ///                   call blocks until the runtime system has been 
        ///                   fully stopped. If this parameter is \a false then 
        ///                   this call will initiate the stop action but will
        ///                   return immediately. Use a second call to stop 
        ///                   with this parameter set to \a true to wait for 
        ///                   all internal work to be completed.
        void stop(bool blocking = true);

        /// \brief Run the HPX runtime system, use the given function for the 
        ///        main \a thread and block waiting for all threads to 
        ///        finish
        ///
        /// \param func       [in] This is the main function of an HPX 
        ///                   application. It will be scheduled for execution
        ///                   by the thread manager as soon as the runtime has 
        ///                   been initialized. This function is expected to 
        ///                   expose an interface as defined by the typedef
        ///                   \a hpx_main_function_type. This parameter is 
        ///                   optional and defaults to none main thread 
        ///                   function, in which case all threads have to be 
        ///                   scheduled explicitly.
        /// \param num_threads [in] The initial number of threads to be started 
        ///                   by the threadmanager. This parameter is optional 
        ///                   and defaults to 1.
        ///
        /// \note             The parameter \a func is optional. If no function
        ///                   is supplied, the runtime system will simply wait
        ///                   for the shutdown action without explicitly 
        ///                   executing any main thread.
        ///
        /// \returns          This function will return the value as returned 
        ///                   as the result of the invocation of the function 
        ///                   object given by the parameter \p func.
        int run(boost::function<hpx_main_function_type> func =
                    boost::function<hpx_main_function_type>(), 
                std::size_t num_threads = 1);

        /// \brief Run the HPX runtime system, initially use the given number 
        ///        of (OS) threads in the threadmanager and block waiting for
        ///        all threads to finish.
        ///
        /// \param num_threads [in] The initial number of threads to be started 
        ///                   by the threadmanager. 
        ///
        /// \returns          This function will always return 0 (zero).
        int run(std::size_t num_threads);

        ///////////////////////////////////////////////////////////////////////

        /// \brief Allow access to the AGAS client instance used by the HPX
        ///        runtime.
        naming::resolver_client const& get_agas_client() const
        {
            return agas_client_;
        }

        /// \brief Allow access to the parcel handler instance used by the HPX
        ///        runtime.
        parcelset::parcelhandler& get_parcel_handler()
        {
            return parcel_handler_;
        }

        /// \brief Allow access to the thread manager instance used by the HPX
        ///        runtime.
        threads::threadmanager& get_thread_manager()
        {
            return thread_manager_;
        }

        /// \brief Allow access to the applier instance used by the HPX
        ///        runtime.
        applier::applier& get_applier()
        {
            return applier_;
        }

        /// \brief Allow access to the action manager instance used by the HPX
        ///        runtime.
        actions::action_manager& get_action_manager()
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
        mode mode_;
        util::runtime_configuration ini_;
        util::io_service_pool agas_pool_; 
        util::io_service_pool parcel_pool_; 
        util::io_service_pool timer_pool_; 
        naming::resolver_client agas_client_;
        parcelset::parcelport parcel_port_;
        threads::threadmanager thread_manager_;
        parcelset::parcelhandler parcel_handler_;
        util::detail::init_logging init_logging_;
        components::server::memory memory_;
        applier::applier applier_;
        actions::action_manager action_manager_;
        components::server::runtime_support runtime_support_;
    };

}   // namespace hpx

#include <hpx/config/warnings_suffix.hpp>

#endif
