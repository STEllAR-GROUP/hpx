//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_REGISTRY_BASE_MAR_10_2010_0710PM)
#define HPX_COMPONENT_REGISTRY_BASE_MAR_10_2010_0710PM

#include <hpx/config.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>
#include <boost/mpl/list.hpp>

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
        virtual bool get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath) = 0;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the components.
#define HPX_REGISTER_COMPONENT_REGISTRY(RegistryType, componentname)          \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                            \
        hpx::components::component_registry_base, RegistryType,               \
        componentname, registry)                                              \
/**/

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_APPLICATION_NAME)
/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_REGISTRY_MODULE()                                        \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, registry)             \
/**/
#else
// in executables (when HPX_APPLICATION_NAME is defined) this needs to expand
// to nothing
#define HPX_REGISTER_REGISTRY_MODULE()
#endif

#endif

