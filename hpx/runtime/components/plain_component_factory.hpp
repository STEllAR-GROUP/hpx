//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file plain_component_factory.hpp

#if !defined(HPX_PLAIN_COMPONENT_FACTORY_JUN_18_2010_1100AM)
#define HPX_PLAIN_COMPONENT_FACTORY_JUN_18_2010_1100AM

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/components/unique_component_name.hpp>
#include <hpx/runtime/components/component_factory_base.hpp>
#include <hpx/runtime/components/component_registry.hpp>
#include <hpx/runtime/components/server/plain_function.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/detail/count_num_args.hpp>

#include <boost/serialization/export.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    /// The \a plain_component_factory provides a minimal implementation of a
    /// component's factory usable for plain_actions.
    ///
    /// \tparam Action   The plain action type this factory should be
    ///                  responsible for.
    template <typename Action>
    struct plain_component_factory : public component_factory_base
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
        plain_component_factory(util::section const*, util::section const*,
                bool isenabled)
          : isenabled_(isenabled)
        {}

        ///
        ~plain_component_factory() {}

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
            typedef server::plain_function<Action> type_holder;
            if (component_invalid == components::get_component_type<type_holder>())
            {
                // First call to get_component_type, ask AGAS for a unique id.
                if (isenabled_) {
                    component_type const ctype =
                        agas_client.register_factory(locality, get_component_name());

                    if (component_invalid == ctype)
                        HPX_THROW_EXCEPTION(duplicate_component_id,
                            "component_factory::get_component_type",
                            "the component name " + get_component_name() +
                            " is already in use");

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
            return unique_component_name<plain_component_factory>::call();
        }

        /// \brief  The function \a get_factory_properties is used to
        ///         determine, whether instances of the derived component can
        ///         be created in blocks (i.e. more than one instance at once).
        ///         This function is used by the \a distributing_factory to
        ///         determine a correct allocation strategy
        factory_property get_factory_properties() const
        {
            return components::factory_none;
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
            HPX_THROW_EXCEPTION(bad_request,
                "plain_component_factory::create",
                "create is not supported by this factory instance (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function.
        ///
        /// \param f  [in] The constructor function to call in order to
        ///         initialize the newly allocated object.
        ///
        /// \return Returns the GID of the first newly created component
        ///         instance. If more than one component instance has been
        ///         created (\a count > 1) the GID's of all new instances are
        ///         sequential in a row.
        naming::gid_type create_with_args(HPX_STD_FUNCTION<void(void*)> const&)
        {
            HPX_THROW_EXCEPTION(bad_request,
                "plain_component_factory::create_with_args",
                "create_with_args is not supported by this factory instance (" +
                get_component_name() + ")");
            return naming::invalid_gid;
        }

        /// \brief Destroy one or more component instances
        ///
        /// \param gid    [in] The gid of the first component instance to
        ///               destroy.
        void destroy(naming::gid_type const& /*gid*/)
        {
        }

        /// \brief Ask how many instances are alive of the type this factory is
        ///        responsible for
        ///
        /// \return Returns the number of instances of the managed object type
        ///         which are currently alive.
        long instance_count() const
        {
            return 1;   // there is always exactly one instance
        }

    protected:
        bool isenabled_;
    };

    /// \endcond
}}

/// \cond NOINTERNAL

///////////////////////////////////////////////////////////////////////////////
// This macro is used to create and to register a minimal factory for plain
// actions with Hpx.Plugin.
#define HPX_REGISTER_PLAIN_ACTION_(...)                                       \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_PLAIN_ACTION_, HPX_UTIL_PP_NARG(__VA_ARGS__)             \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_PLAIN_ACTION_1(action_type)                              \
    HPX_REGISTER_PLAIN_ACTION_3(action_type, action_type,                     \
        ::hpx::components::factory_check)                                     \
/**/

#define HPX_REGISTER_PLAIN_ACTION_2(action_type, plain_action_name)           \
    HPX_REGISTER_PLAIN_ACTION_3(action_type, plain_action_name,               \
        ::hpx::components::factory_check)                                     \
/**/

#define HPX_REGISTER_PLAIN_ACTION_3(action_type, plain_action_name, state)    \
    BOOST_CLASS_EXPORT_KEY2(hpx::actions::transfer_action<action_type>,       \
        BOOST_PP_STRINGIZE(plain_action_name))                                \
    HPX_REGISTER_ACTION_2(action_type, plain_action_name)                     \
    HPX_REGISTER_COMPONENT_FACTORY(                                           \
        hpx::components::plain_component_factory<action_type>,                \
        plain_action_name)                                                    \
    HPX_DEF_UNIQUE_COMPONENT_NAME(                                            \
        hpx::components::plain_component_factory<action_type>,                \
        plain_action_name)                                                    \
    template struct hpx::components::plain_component_factory<action_type>;    \
    HPX_REGISTER_MINIMAL_COMPONENT_REGISTRY_3(                                \
        hpx::components::server::plain_function<action_type>,                 \
        plain_action_name, state)                                             \
    HPX_DEFINE_GET_COMPONENT_TYPE(                                            \
        hpx::components::server::plain_function<action_type>)                 \
/**/

/// \endcond

#endif

