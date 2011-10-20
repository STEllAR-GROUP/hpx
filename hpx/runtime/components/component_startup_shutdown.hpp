//  Copyright (c) 2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_STARTUP_SHUTDOWN_SEP_20_2011_0217PM)
#define HPX_COMPONENT_STARTUP_SHUTDOWN_SEP_20_2011_0217PM

#include <boost/config.hpp>
#include <boost/function.hpp>

#include <hpx/runtime/components/component_startup_shutdown_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace startup_shutdown_provider
    {
        bool startup(boost::function<void()>& startup_func);
        bool shutdown(boost::function<void()>& shutdown_func);
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \class component_startup_shutdown component_startup_shutdown.hpp hpx/runtime/components/component_startup_shutdown.hpp
    ///
    /// The \a component_startup_shutdown provides a minimal implementation of
    /// a component's startup/shutdown function provider.
    ///
    struct component_startup_shutdown : public component_startup_shutdown_base
    {
        ///
        ~component_startup_shutdown() {}

        /// \brief Return any startup function for this component
        ///
        /// \param startup  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a startup has been
        ///         successfully initialized with the startup function.
        bool get_startup_function(boost::function<void()>& startup_)
        {
            return startup_shutdown_provider::startup(startup_);
        }

        /// \brief Return any startup function for this component
        ///
        /// \param shutdown  [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a shutdown has been
        ///         successfully initialized with the shutdown function.
        bool get_shutdown_function(boost::function<void()>& shutdown_)
        {
            return startup_shutdown_provider::shutdown(shutdown_);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEF_COMPONENT_STARTUP_SHUTDOWN(startup_, shutdown_)               \
    namespace hpx { namespace components { namespace startup_shutdown_provider\
    {                                                                         \
        bool startup(boost::function<void()>& startup_func)                   \
        {                                                                     \
            boost::function<void()> tmp = startup_;                           \
            if (!tmp.empty()) { startup_func = startup_; return true; }       \
            return false;                                                     \
        }                                                                     \
        bool shutdown(boost::function<void()>& shutdown_func)                 \
        {                                                                     \
            boost::function<void()> tmp = shutdown_;                          \
            if (!tmp.empty()) { shutdown_func = shutdown_; return true; }     \
            return false;                                                     \
        }                                                                     \
    }}}                                                                       \
    /***/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(startup, shutdown)               \
        HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                             \
        HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(                               \
            hpx::components::component_startup_shutdown, startup_shutdown)    \
        HPX_DEF_COMPONENT_STARTUP_SHUTDOWN(startup, shutdown)                 \
    /**/

#endif // HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649

