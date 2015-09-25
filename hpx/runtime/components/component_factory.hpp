//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/components/component_factory.hpp

#if !defined(HPX_COMPONENT_FACTORY_SEP_26_2008_0647PM)
#define HPX_COMPONENT_FACTORY_SEP_26_2008_0647PM

#ifdef DOXYGEN
/// \def HPX_REGISTER_COMPONENT(type, name, mode)
///
/// \brief Define a component factory for a component type
///
/// This macro is used create and to register a minimal component factory for
/// a component type which allows it to be remotely created using the
/// \a hpx::new_<> function.
///
/// This macro can be invoked with one, two or three arguments
///
/// \param type The \a type parameter is a (fully decorated) type of the
///             component type for which a factory should be defined.
///
/// \param name The \a name parameter specifies the name to use to register
///             the factory. This should uniquely (system-wide) identify the
///             component type. The \a name parameter must conform to the C++
///             identifier rules (without any namespace).
///             If this parameter is not given, the first parameter is used.
///
/// \param mode The \a mode parameter has to be one of the defined enumeration
///             values of the enumeration \a hpx::components::factory_state_enum.
///             The default for this parameter is
///             \a hpx::components::factory_enabled.
///
#define HPX_REGISTER_COMPONENT(type, name, mode)

#else

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/unique_function.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_factory provides a minimal implementation of a
    /// component's factory. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required
    /// functions to be exposed by a component's factory instance.
    ///
    /// \tparam Component   The component type this factory should be
    ///                     responsible for.
    template <typename Component>
    struct component_factory : public component_factory_base
    {
        /// \brief Construct a new factory instance
        ///
        /// \param global   [in] The pointer to a \a hpx#util#section instance
        ///                 referencing the settings read from the [settings]
        ///                 section of the global configuration file (hpx.ini)
        ///                 This pointer may be NULL if no such section has
        ///                 been found.
        /// \param local    [in] The pointer to a \a hpx#util#section instance
        ///                 referencing the settings read from the section
        ///                 describing this component type:
        ///                 [hpx.components.\<name\>], where \<name\> is the
        ///                 instance name of the component as given in the
        ///                 configuration files.
        ///
        /// \note The contents of both sections has to be cloned in order to
        ///       save the configuration setting for later use.
        component_factory(util::section const* global,
                util::section const* local, bool isenabled)
          : isenabled_(isenabled), refcnt_(0)
        {
            // store the configuration settings
            if (NULL != global)
                global_settings_ = *global;
            if (NULL != local)
                local_settings_ = *local;
        }

        /// Destruct a \a component_factory instance
        ~component_factory() {}

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
        component_type get_component_type(naming::gid_type const& locality,
            naming::resolver_client& agas_client)
        {
            typedef typename Component::type_holder type_holder;
            if (component_invalid == components::get_component_type<type_holder>())
            {
                // First call to get_component_type, ask AGAS for a unique id.
                if (isenabled_) {
                    component_type const ctype =
                        agas_client.register_factory(locality, get_component_name());

                    if (component_invalid == ctype) {
                        HPX_THROW_EXCEPTION(duplicate_component_id,
                            "component_factory::get_component_type",
                            "the component name " + get_component_name() +
                            " is already in use");
                    }
                    components::set_component_type<type_holder>(ctype);
                }
                else {
                    component_type const ctype =
                        agas_client.get_component_id(get_component_name());

                    components::set_component_type<type_holder>(ctype);
                }
            }
            return components::get_component_type<type_holder>();
        }

        /// \brief Return the name of the component type this factory is
        ///        responsible for
        ///
        /// \return Returns the name of the component type this factory
        ///         instance is responsible for. This function throws on any
        ///         error.
        std::string get_component_name() const
        {
            return unique_component_name<component_factory>::call();
        }

        /// \brief Create one or more new component instances.
        ///
        /// \param count  [in] The number of component instances to
        ///               create. The value of this parameter should not
        ///               be zero.
        ///
        /// \return Returns the GID of the first newly created component
        ///         instance. If more than one component instance has been
        ///         created (\a count > 1) the GID's of all new instances are
        ///         sequential in a row.
        naming::gid_type create(std::size_t count = 1)
        {
            if (isenabled_)
            {
                naming::gid_type id = server::create<Component>(count);
                if (id)
                    ++refcnt_;
                return id;
            }

            HPX_THROW_EXCEPTION(bad_request,
                "component_factory::create",
                "this factory instance is disabled for this locality (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function.
        ///
        /// \param ctor  [in] The constructor function to call in order to
        ///              initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance.
        naming::gid_type create_with_args(
            util::unique_function_nonser<void(void*)> const& ctor)
        {
            if (isenabled_)
            {
                naming::gid_type id = server::create<Component>(ctor);
                if (id)
                    ++refcnt_;
                return id;
            }

            HPX_THROW_EXCEPTION(bad_request,
                "component_factory::create_with_args",
                "this factory instance is disabled for this locality (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function. Assign the give
        ///        GID to the new object.
        ///
        /// \param assign_gid [in] The GID to assign to the newly created object.
        /// \param ctor       [in] The constructor function to call in order to
        ///                   initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance (this is the same as assign_gid, if successful).
        naming::gid_type create_with_args(
            naming::gid_type const& assign_gid,
            util::unique_function_nonser<void(void*)> const& ctor)
        {
            if (isenabled_)
            {
                naming::gid_type id =
                    server::create<Component>(assign_gid, ctor);
                if (id)
                    ++refcnt_;
                return id;
            }

            HPX_THROW_EXCEPTION(bad_request,
                "component_factory::create_with_args",
                "this factory instance is disabled for this locality (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Destroy one or more component instances
        ///
        /// \param gid    [in] The gid of the first component instance to
        ///               destroy.
        /// \param addr   [in] The resolved address of the first component
        ///               instance to destroy.
        void destroy(naming::gid_type const& gid,
            naming::address const& addr)
        {
            server::destroy<Component>(gid, addr);
            --refcnt_;
        }

        /// \brief Ask how many instances are alive of the type this factory is
        ///        responsible for
        ///
        /// \return Returns the number of instances of the managed object type
        ///         which are currently alive.
        long instance_count() const
        {
            return refcnt_;
        }

#if defined(HPX_HAVE_SECURITY)
        /// \brief Return the required capabilities necessary to create an
        ///        instance of a component using this factory instance.
        ///
        /// \return Returns required capabilities necessary to create a new
        ///         instance of a component using this factory instance.
        virtual components::security::capability
            get_required_capabilities() const
        {
            using namespace components::security;
            return Component::get_required_capabilities(
                traits::capability<>::capability_create_component);
        }
#endif

    protected:
        util::section global_settings_;
        util::section local_settings_;
        bool isenabled_;

        // count outstanding instances to avoid premature unloading
        boost::detail::atomic_count refcnt_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
// This macro is used create and to register a minimal component factory with
// Hpx.Plugin.
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(...)                           \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_COMPONENT(...)                                           \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY(ComponentType, componentname)  \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                             \
            ComponentType, componentname, ::hpx::components::factory_enabled) \
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/

#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY(ComponentType, componentname) \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                             \
            ComponentType, componentname, ::hpx::components::factory_disabled)\
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(...)                          \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_, HPX_UTIL_PP_NARG(__VA_ARGS__)\
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_1(ComponentType)               \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, ComponentType, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_2(ComponentType, componentname)\
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                                 \
        ComponentType, componentname, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_3(                             \
        ComponentType, componentname, state)                                  \
    HPX_REGISTER_COMPONENT_FACTORY(                                           \
        hpx::components::component_factory<ComponentType>, componentname)     \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        hpx::components::component_factory<ComponentType>, componentname)     \
    template struct hpx::components::component_factory<ComponentType>;        \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                \
        ComponentType, componentname, state)                                  \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC(...)                   \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(__VA_ARGS__)              \
/**/

// same as above, just a better name

/// This macro is used create and to register a minimal component factory for
/// a component type which allows it to be remotely created using the
/// hpx::new_<> function.
/// This macro can be invoked with one, two or three arguments
#define HPX_REGISTER_COMPONENT_DYNAMIC(...)                                   \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(__VA_ARGS__)              \
/**/

#define HPX_REGISTER_ENABLED_COMPONENT_FACTORY_DYNAMIC(ComponentType,         \
            componentname)                                                    \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
            ComponentType, componentname, ::hpx::components::factory_enabled) \
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/

#define HPX_REGISTER_DISABLED_COMPONENT_FACTORY_DYNAMIC(ComponentType,        \
            componentname)                                                    \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
            ComponentType, componentname, ::hpx::components::factory_disabled)\
        HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)            \
/**/


#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_(...)                  \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_,                      \
            HPX_UTIL_PP_NARG(__VA_ARGS__)                                     \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_1(ComponentType)       \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                         \
        ComponentType, ComponentType, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_2(ComponentType,       \
        componentname)                                                        \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                         \
        ComponentType, componentname, ::hpx::components::factory_check)       \
    HPX_DEFINE_GET_COMPONENT_TYPE(ComponentType::wrapped_type)                \
/**/
#define HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_DYNAMIC_3(                     \
        ComponentType, componentname, state)                                  \
    HPX_REGISTER_COMPONENT_FACTORY_DYNAMIC(                                   \
        hpx::components::component_factory<ComponentType>, componentname)     \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        hpx::components::component_factory<ComponentType>, componentname)     \
    template struct hpx::components::component_factory<ComponentType>;        \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_DYNAMIC_3(                        \
        ComponentType, componentname, state)                                  \
/**/

#endif

#endif
