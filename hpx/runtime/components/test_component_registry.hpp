//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TEST_COMPONENT_REGISTRY_FEB_21_2012_1046AM)
#define HPX_TEST_COMPONENT_REGISTRY_FEB_21_2012_1046AM

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
    /// The \a test_component_registry provides a minimal implementation of a
    /// component's registry. It is special in the sense that it disables the
    /// component by default, requiring the application to explicitly enable it
    /// when needed
    ///
    /// \tparam Component   The component type this registry should be
    ///                     responsible for.
    template <typename Component>
    struct test_component_registry : public component_registry_base
    {
        ///
        ~test_component_registry() {}

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
                unique_component_name<test_component_registry>::call() + "]";
            fillini += "name = " HPX_COMPONENT_STRING;
            fillini += "path = $[hpx.location]/lib/hpx/" HPX_LIBRARY;
            fillini += "enabled = 0";
            return true;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY is used create and to
/// register a minimal component registry with Boost.Plugin.
#define HPX_REGISTER_TEST_COMPONENT_REGISTRY(ComponentType, componentname)    \
        typedef hpx::components::test_component_registry<ComponentType>       \
            componentname ## _test_component_registry_type;                   \
        HPX_REGISTER_COMPONENT_REGISTRY(                                      \
            componentname ## _test_component_registry_type,                   \
            componentname)                                                    \
        HPX_DEF_UNIQUE_COMPONENT_NAME(                                        \
            componentname ## _test_component_registry_type,                   \
            componentname)                                                    \
        template struct hpx::components::test_component_registry<             \
            ComponentType>;                                                   \
    /**/

#endif
