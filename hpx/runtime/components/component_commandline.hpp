//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_COMMANDLINE_JAN_09_2012_1130AM)
#define HPX_COMPONENT_COMMANDLINE_JAN_09_2012_1130AM

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_commandline_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    namespace commandline_options_provider
    {
        boost::program_options::options_description add_commandline_options();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_startup_shutdown provides a minimal implementation of
    /// a component's startup/shutdown function provider.
    ///
    struct component_commandline : public component_commandline_base
    {
        ///
        ~component_commandline() {}

        /// \brief Return any additional command line options valid for this
        ///        component
        ///
        /// \return The module is expected to fill a options_description object
        ///         with any additional command line options this component
        ///         will handle.
        ///
        /// \note   This function will be executed by the runtime system
        ///         during system startup.
        boost::program_options::options_description add_commandline_options()
        {
            return commandline_options_provider::add_commandline_options();
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)        \
    namespace hpx { namespace components { namespace commandline_options_provider \
    {                                                                         \
        boost::program_options::options_description add_commandline_options() \
        {                                                                     \
            return add_options_function();                                    \
        }                                                                     \
    }}}                                                                       \
    /***/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_COMMANDLINE_MODULE(add_options_function)                 \
        HPX_REGISTER_COMMANDLINE_OPTIONS()                                    \
        HPX_REGISTER_COMMANDLINE_REGISTRY(                                    \
            hpx::components::component_commandline, commandline_options)      \
        HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)        \
    /**/
#define HPX_REGISTER_COMMANDLINE_MODULE_DYNAMIC(add_options_function)         \
        HPX_REGISTER_COMMANDLINE_OPTIONS_DYNAMIC()                            \
        HPX_REGISTER_COMMANDLINE_REGISTRY_DYNAMIC(                            \
            hpx::components::component_commandline, commandline_options)      \
        HPX_DEFINE_COMPONENT_COMMANDLINE_OPTIONS(add_options_function)        \
    /**/

#endif // HPX_A7F46A4F_9AF9_4909_B0D8_5304FEFC5649

