//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_STARTUP_SHUTDOWN_SEP_20_2011_0217PM)
#define HPX_COMPONENT_STARTUP_SHUTDOWN_SEP_20_2011_0217PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_startup_shutdown_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_startup_shutdown class provides a minimal
    /// implementation of a component's startup/shutdown function provider.
    template <bool(*Startup)(HPX_STD_FUNCTION<void()>&),
        bool(*Shutdown)(HPX_STD_FUNCTION<void()>&)>    
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
        bool get_startup_function(HPX_STD_FUNCTION<void()>& startup_)
        {
            return Startup(startup_);
        }

        /// \brief Return any startup function for this component
        ///
        /// \param shutdown [in, out] The module is expected to fill this
        ///                 function object with a reference to a startup
        ///                 function. This function will be executed by the
        ///                 runtime system during system startup.
        ///
        /// \return Returns \a true if the parameter \a shutdown has been
        ///         successfully initialized with the shutdown function.
        bool get_shutdown_function(HPX_STD_FUNCTION<void()>& shutdown_)
        {
            return Shutdown(shutdown_);
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_COMPONENT_STARTUP_SHUTDOWN(startup_, shutdown_)            \
    namespace hpx { namespace components { namespace startup_shutdown_provider\
    {                                                                         \
        bool BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _startup)                   \
            (HPX_STD_FUNCTION<void()>& startup_func)                          \
        {                                                                     \
            HPX_STD_FUNCTION<bool(HPX_STD_FUNCTION<void()>&)> tmp = startup_; \
            if (!!tmp) { return tmp(startup_func); }                          \
            return false;                                                     \
        }                                                                     \
        bool BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _shutdown)                  \
            (HPX_STD_FUNCTION<void()>& shutdown_func)                         \
        {                                                                     \
            HPX_STD_FUNCTION<bool(HPX_STD_FUNCTION<void()>&)> tmp = shutdown_;\
            if (!!tmp) { return tmp(shutdown_func); }                         \
            return false;                                                     \
        }                                                                     \
    }}}                                                                       \
    /***/

#define HPX_NULL_STARTUP_SHUTDOWN_PTR ((bool(*)(HPX_STD_FUNCTION<void()>&))0)

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(startup, shutdown)               \
    HPX_REGISTER_STARTUP_SHUTDOWN_FUNCTIONS()                                 \
    HPX_DEFINE_COMPONENT_STARTUP_SHUTDOWN(startup, shutdown)                  \
    namespace hpx { namespace components { namespace startup_shutdown_provider\
    {                                                                         \
        typedef component_startup_shutdown<                                   \
            BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _startup),                   \
            BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _shutdown)                   \
        > BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _provider);                    \
    }                                                                         \
        template struct component_startup_shutdown<                           \
            startup_shutdown_provider::                                       \
                BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _startup),               \
            startup_shutdown_provider::                                       \
                BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _shutdown)               \
        >;                                                                    \
    }}                                                                        \
    HPX_REGISTER_STARTUP_SHUTDOWN_REGISTRY(                                   \
        hpx::components::startup_shutdown_provider::                          \
        BOOST_PP_CAT(HPX_COMPONENT_LIB_NAME, _provider), startup_shutdown)    \
    /**/

#define HPX_REGISTER_STARTUP_MODULE(startup)                                  \
        HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(                                 \
            startup, HPX_NULL_STARTUP_SHUTDOWN_PTR)                           \
    /**/

#endif // HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649

