//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM)
#define HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class component_registry component_registry.hpp hpx/runtime/components/component_registry.hpp
    ///
    /// The \a component_registry provides a minimal implementation of a 
    /// component's registry. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required 
    /// functions to be exposed by a component's registry instance.
    ///
    /// \tparam Component   The component type this registry should be 
    ///                     responsible for.
    template <typename Component>
    struct component_registry : public component_registry_base
    {
        ///
        ~component_registry() {}

        /// \brief Return the ini-information for all contained components
        ///
        /// \param fillini  [in] The module is expected to fill this vector 
        ///                 with the ini-information (one line per vector 
        ///                 element) for all components implemented in this
        ///                 module.
        ///
        /// \return Returns \a true if the parameter \a fillini has been
        ///         successfully initialized with the registry data of all 
        ///         implemented in this module.
        bool get_component_info(std::vector<std::string>& fillini)
        {
            using namespace boost::assign;
            fillini += std::string("[hpx.components.") + 
                unique_component_name<component_registry>::call() + "]";
            fillini += "name = " HPX_COMPONENT_STRING;
            fillini += "path = $[hpx.location]/lib/hpx/" HPX_LIBRARY;
            fillini += "enabled = $[hpx.components.load_external]";
            return true;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY is used create and to 
/// register a minimal component registry with Boost.Plugin. 
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY(ComponentType, componentname) \
        HPX_REGISTER_COMPONENT_REGISTRY(                                      \
            hpx::components::component_registry<ComponentType>,               \
            componentname);                                                   \
        HPX_DEF_UNIQUE_COMPONENT_NAME(                                        \
            hpx::components::component_registry<ComponentType>,               \
            componentname)                                                    \
        template struct hpx::components::component_registry<ComponentType>;   \
    /**/

#endif
