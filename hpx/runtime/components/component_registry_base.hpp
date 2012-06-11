//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_REGISTRY_BASE_MAR_10_2010_0710PM)
#define HPX_COMPONENT_REGISTRY_BASE_MAR_10_2010_0710PM

#include <boost/plugin.hpp>
#include <boost/plugin/export_plugin.hpp>
#include <boost/mpl/list.hpp>

#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
// FIXME: Can we move this to config.hpp?
#if !defined(HPX_COMPONENT_LIB_NAME)
#define HPX_COMPONENT_LIB_NAME                                                \
        HPX_MANGLE_COMPONENT_NAME(HPX_COMPONENT_NAME)                         \
    /**/
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_registry_base has to be used as a base class for all
    /// component registries.
    struct HPX_EXPORT component_registry_base
    {
        virtual ~component_registry_base() {}

        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in, out] The module is expected to fill this vector
        ///                 with the ini-information (one line per vector
        ///                 element) for all components implemented in this
        ///                 module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all
        ///         implemented in this module.
        virtual bool get_component_info(std::vector<std::string>& fillini) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Boost.Plugin. This macro has to be used for each of the components.
#define HPX_REGISTER_COMPONENT_REGISTRY(RegistryType, componentname)          \
        BOOST_PLUGIN_EXPORT(HPX_COMPONENT_LIB_NAME,                           \
            hpx::components::component_registry_base, RegistryType,           \
            componentname, HPX_MANGLE_COMPONENT_NAME(registry))               \
    /**/

/// This macro is used to define the required Boost.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_REGISTRY_MODULE()                                        \
        BOOST_PLUGIN_EXPORT_LIST(HPX_COMPONENT_LIB_NAME,                      \
            HPX_MANGLE_COMPONENT_NAME(registry))                              \
    /**/

#endif

