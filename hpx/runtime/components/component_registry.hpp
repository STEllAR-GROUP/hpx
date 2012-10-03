//  Copyright (c) 2007-2012 Hartmut Kaiser
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
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_registry provides a minimal implementation of a
    /// component's registry. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required
    /// functions to be exposed by a component's registry instance.
    ///
    /// \tparam Component   The component type this registry should be
    ///                     responsible for.
    template <typename Component, factory_state_enum state>
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
            fillini += "path = $[hpx.location]/lib/hpx/";
            switch (state)
            {
                case factory_enabled:
                    fillini += "enabled = 1";
                    break;
                case factory_disabled:
                    fillini += "enabled = 0";
                    break;
                case factory_check:
                    fillini += "enabled = $[hpx.components.load_external]";
                    break;
            };
            return true;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component registry with
/// Boost.Plugin.

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY(...)                          \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(__VA_ARGS__)                 \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(...)                         \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_2(                            \
        ComponentType, componentname)                                         \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                \
        ComponentType, componentname, ::hpx::components::factory_check)       \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                            \
        ComponentType, componentname, state)                                  \
    typedef hpx::components::component_registry<ComponentType, state>         \
        componentname ## _component_registry_type;                            \
    HPX_REGISTER_COMPONENT_REGISTRY(                                          \
        componentname ## _component_registry_type, componentname)             \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        componentname ## _component_registry_type, componentname)             \
    template struct hpx::components::component_registry<                      \
        ComponentType, state>;                                                \
/**/

#endif

