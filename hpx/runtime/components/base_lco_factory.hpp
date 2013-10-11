//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_BASE_LCO_FACTORY_OCT_10_2013_1118AM)
#define HPX_BASE_LCO_FACTORY_OCT_10_2013_1118AM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/server/create_component.hpp>
#include <hpx/runtime/components/server/destroy_component.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a base_lco_factory provides a special implementation of a
    /// component factory for components exposing the base_lco interface.
    ///
    /// \tparam Component   The component type this factory should be
    ///                     responsible for.
    template <typename Component>
    struct base_lco_factory : public component_factory_base
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
        base_lco_factory(util::section const* global,
                util::section const* local, bool isenabled)
          : isenabled_(isenabled), refcnt_(0)
        {
            // store the configuration settings
            if (NULL != global)
                global_settings_ = *global;
            if (NULL != local)
                local_settings_ = *local;
        }

        ///
        ~base_lco_factory() {}

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
            BOOST_ASSERT(components::component_invalid != 
                components::get_component_type<type_holder>());
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
            return unique_component_name<base_lco_factory>::call();
        }

        /// \brief  The function \a get_factory_properties is used to
        ///         determine, whether instances of the derived component can
        ///         be created in blocks (i.e. more than one instance at once).
        ///         This function is used by the \a distributing_factory to
        ///         determine a correct allocation strategy
        factory_property get_factory_properties() const
        {
            return factory_none;
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
                "base_lco_factory::create",
                "this factory instance is disabled for this locality (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function.
        ///
        /// \param f  [in] The constructor function to call in order to
        ///           initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance.
        naming::gid_type create_with_args(HPX_STD_FUNCTION<void(void*)> const& ctor)
        {
            if (isenabled_)
            {
                naming::gid_type id = server::create<Component>(ctor);
                if (id)
                    ++refcnt_;
                return id;
            }

            HPX_THROW_EXCEPTION(bad_request,
                "base_lco_factory::create_with_args",
                "this factory instance is disabled for this locality (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Destroy one or more component instances
        ///
        /// \param gid    [in] The gid of the first component instance to
        ///               destroy.
        void destroy(naming::gid_type const& gid)
        {
            server::destroy<Component>(gid);
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
/// This macro is used create and to register a base_lco factory with Hpx.Plugin.
#define HPX_REGISTER_BASE_LCO_FACTORY(...)                                    \
    HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_(__VA_ARGS__)                      \
/**/

#define HPX_REGISTER_ENABLED_BASE_LCO_FACTORY(ComponentType, componentname)   \
        HPX_REGISTER_BASE_LCO_FACTORY_3(                                      \
            ComponentType, componentname, ::hpx::components::factory_enabled) \
/**/

#define HPX_REGISTER_DISABLED_BASE_LCO_FACTORY(ComponentType, componentname)  \
        HPX_REGISTER_BASE_LCO_FACTORY_3(                                      \
            ComponentType, componentname, ::hpx::components::factory_disabled)\
/**/


#define HPX_REGISTER_BASE_LCO_FACTORY_(...)                                   \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_BASE_LCO_FACTORY_, HPX_UTIL_PP_NARG(__VA_ARGS__)         \
    )(__VA_ARGS__))                                                           \
/**/
#define HPX_REGISTER_BASE_LCO_FACTORY_2(ComponentType, componentname)         \
    HPX_REGISTER_BASE_LCO_FACTORY_3(                                          \
        ComponentType, componentname, ::hpx::components::factory_check)       \
/**/
#define HPX_REGISTER_BASE_LCO_FACTORY_3(                                      \
        ComponentType, componentname, state)                                  \
    HPX_REGISTER_COMPONENT_FACTORY(                                           \
        hpx::components::base_lco_factory<ComponentType>, componentname)      \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        hpx::components::base_lco_factory<ComponentType>, componentname)      \
    template struct hpx::components::base_lco_factory<ComponentType>;         \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                \
        ComponentType, componentname, state)                                  \
/**/

#endif

