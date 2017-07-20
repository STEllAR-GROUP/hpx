//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#if !defined(HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM)
#define HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM

#include <hpx/config.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/util/detail/pp/cat.hpp>
#include <hpx/util/detail/pp/nargs.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/find_prefix.hpp>

#include <hpx/traits/component_config_data.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/assign/std/vector.hpp>

#include <string>
#include <vector>

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
        bool get_component_info(std::vector<std::string>& fillini,
            std::string const& filepath, bool is_static = false)
        {
            using namespace boost::assign;
            fillini += std::string("[hpx.components.") +
                unique_component_name<component_registry>::call() + "]";
            fillini += "name = " HPX_COMPONENT_STRING;

            if(!is_static)
            {
                if (filepath.empty()) {
                    fillini += std::string("path = ") +
                        util::find_prefixes("/hpx", HPX_COMPONENT_STRING);
                }
                else {
                    fillini += std::string("path = ") + filepath;
                }
            }

            switch (state) {
            case factory_enabled:
                fillini += "enabled = 1";
                break;
            case factory_disabled:
                fillini += "enabled = 0";
                break;
            case factory_check:
                fillini += "enabled = $[hpx.components.load_external]";
                break;
            }

            if (is_static) {
                fillini += "static = 1";
            }

            char const* more = traits::component_config_data<Component>::call();
            if (more) {
                std::vector<std::string> data;
                boost::split(data, more, boost::is_any_of("\n"));
                std::copy(data.begin(), data.end(), std::back_inserter(fillini));
            }
            return true;
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used create and to register a minimal component registry with
/// Hpx.Plugin.

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY(...)                          \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(__VA_ARGS__)                 \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_(...)                         \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_, HPX_PP_NARGS(__VA_ARGS__)   \
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

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC(...)                  \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_(__VA_ARGS__)         \
    /**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_(...)                 \
    HPX_PP_EXPAND(HPX_PP_CAT(                                                 \
        HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_,                     \
            HPX_PP_NARGS(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_2(                    \
        ComponentType, componentname)                                         \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                        \
        ComponentType, componentname, ::hpx::components::factory_check)       \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                    \
        ComponentType, componentname, state)                                  \
    typedef hpx::components::component_registry<ComponentType, state>         \
        componentname ## _component_registry_type;                            \
    HPX_REGISTER_COMPONENT_REGISTRY_DYNAMIC(                                  \
        componentname ## _component_registry_type, componentname)             \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        componentname ## _component_registry_type, componentname)             \
    template struct hpx::components::component_registry<                      \
        ComponentType, state>;                                                \
/**/

#endif

