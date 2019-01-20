//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Thomas Heller
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#if !defined(HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM)
#define HPX_COMPONENT_REGISTRY_MAR_10_2010_0720PM

#include <hpx/config.hpp>
#include <hpx/pp/cat.hpp>
#include <hpx/pp/expand.hpp>
#include <hpx/pp/nargs.hpp>
#include <hpx/pp/stringize.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/util/find_prefix.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>

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
            typedef typename Component::type_holder type_holder;
            using namespace boost::assign;
            fillini += std::string("[hpx.components.") +
                get_component_name<type_holder>() + "]";
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

        /// \brief Return the unique identifier of the component type this
        ///        factory is responsible for
        ///
        /// \param locality     [in] The id of the locality this factory
        ///                     is responsible for.
        /// \param agas_client  [in] The AGAS client to use for component id
        ///                     registration (if needed).
        ///
        /// \return Returns the unique identifier of the component type this
        ///         factory instance is responsible for. This function throws
        ///         on any error.
        void register_component_type()
        {
            typedef typename Component::type_holder type_holder;

            char const* name = components::get_component_name<type_holder>();
            bool enabled = true;
            hpx::util::runtime_configuration const& config = hpx::get_config();
            std::string enabled_entry = config.get_entry(
                std::string("hpx.components.") + name + ".enabled", "0");

            boost::algorithm::to_lower (enabled_entry);
            if (enabled_entry == "no" || enabled_entry == "false" ||
                enabled_entry == "0")
            {
                LRT_(info) << "plugin factory disabled: " << name;
                enabled = false;     // this component has been disabled
            }

            component_type type = components::get_component_type<type_holder>();
            typedef typename Component::base_type_holder base_type_holder;
            component_type base_type = components::get_component_type<base_type_holder>();
            if (component_invalid == type)
            {
                // First call to get_component_type, ask AGAS for a unique id.
                type = detail::get_agas_component_type(name,
                    components::get_component_base_name<type_holder>(),
                    base_type,
                    enabled
                );
                components::set_component_type<type_holder>(type);
            }
            components::enabled(type) = enabled;
            components::deleter(type) = &server::destroy<Component>;
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
    template struct hpx::components::component_registry<                      \
        ComponentType, state>;                                                \
/**/

#endif

