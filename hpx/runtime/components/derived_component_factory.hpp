//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DERIVED_COMPONENT_FACTORY_NOV_05_2008_1209PM)
#define HPX_DERIVED_COMPONENT_FACTORY_NOV_05_2008_1209PM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/util/ini.hpp>

#include <boost/preprocessor/stringize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class derived_component_factory derived_component_factory.hpp hpx/runtime/components/derived_component_factory.hpp
    ///
    /// The \a derived_component_factory provides a minimal implementation of a 
    /// component's factory. If no additional functionality is required this
    /// type can be used to implement the full set of minimally required 
    /// functions to be exposed by a component's factory instance.
    /// 
    /// The difference to a plain \a component_factory is that it should be 
    /// used for components which are derived from component base classes 
    /// exposing the actions and which dispatch the action functions through
    /// virtual functions to the derived component.
    ///
    /// This is necessary because the lower part of the component type needs to 
    /// match the component type of the component exposing the actions.
    ///
    /// \tparam Component   The component type this factory should be 
    ///                     responsible for.
    template <typename Component>
    struct derived_component_factory : public component_factory_base
    {
        /// 
        static char const* const unique_base_component_name;
        static char const* const unique_component_name;

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
        derived_component_factory(util::section const* global, 
            util::section const* local)
        {
            // store the configuration settings
            if (NULL != global)
                global_settings_ = global->clone();
            if (NULL != local)
                local_settings_ = local->clone();
        }

        ///
        ~derived_component_factory() {}

        /// \brief Return the unique identifier of the component type this 
        ///        factory is responsible for
        ///
        /// \param agas_client  [in] The AGAS client to use for component id 
        ///                     registration (if needed).
        ///
        /// \return Returns the unique identifier of the component type this 
        ///         factory instance is responsible for. This function throws
        ///         on any error.
        component_type get_component_type(naming::resolver_client& agas_client)
        {
            if (component_invalid == Component::get_component_type()) {
                // first call to get_component_type, ask AGAS for a unique id
                Component::set_component_type(derived_component_type(
                    agas_client.get_component_id(unique_component_name),
                    agas_client.get_component_id(unique_base_component_name)));
            }
            return Component::get_component_type();
        }

        /// \brief Return the name of the component type this factory is 
        ///        responsible for
        ///
        /// \return Returns the name of the component type this factory 
        ///         instance is responsible for. This function throws on any 
        ///         error.
        std::string get_component_name() const
        {
            return unique_component_name;
        }

        /// \brief  The function \a has_multi_instance_factory is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        bool has_multi_instance_factory() const
        {
            return Component::has_multi_instance_factory();
        }

        /// \brief Create one or more new component instances.
        ///
        /// \param appl         [in] The applier instance to be used to create
        ///                     the new component instances.
        /// \param count        [in] The number of component instances to 
        ///                     create. The value of this parameter should not 
        ///                     be zero.
        ///
        /// \return Returns the GID of the first newly created component 
        ///         instance. If more than one component instance has been 
        ///         created (\a count > 1) the GID's of all new instances are
        ///         sequential in a row.
        naming::id_type create (applier::applier& appl, std::size_t count)
        {
            return server::create<Component>(appl, count);
        }

        /// \brief Destroy one or more component instances
        ///
        /// \param appl         [in] The applier instance to be used to destroy
        ///                     the component instances.
        /// \param gid          [in] The gid of the first component instance to 
        ///                     destroy. 
        void destroy(applier::applier& appl, naming::id_type const& gid)
        {
            server::destroy<Component>(appl, gid);
        }

    protected:
        util::section global_settings_;
        util::section local_settings_;
    };

}}

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_MINIMAL_COMPONENT_FACTORY is used create and to 
/// register a minimal component factory with Boost.Plugin. This macro may be 
/// used if the registered component factory is the only factory to be exposed 
/// from a particular module. If more than one factories need to be exposed
/// the \a HPX_REGISTER_COMPONENT_FACTORY and \a HPX_REGISTER_COMPONENT_MODULE
/// macros should be used instead.
#define HPX_REGISTER_DERIVED_COMPONENT_FACTORY(ComponentType, componentname,  \
    basecomponentname)                                                        \
        HPX_REGISTER_COMPONENT_FACTORY(                                       \
            hpx::components::derived_component_factory<ComponentType>,        \
            componentname);                                                   \
        template struct                                                       \
            hpx::components::derived_component_factory<ComponentType>;        \
        template<> char const* const                                          \
            hpx::components::derived_component_factory<ComponentType>::       \
                unique_component_name = BOOST_PP_STRINGIZE(componentname);    \
        template<> char const* const                                          \
            hpx::components::derived_component_factory<ComponentType>::       \
                unique_base_component_name = basecomponentname;               \
    /**/

#endif
