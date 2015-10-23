//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_FACTORY_BASE_SEP_26_2008_0446PM)
#define HPX_COMPONENT_FACTORY_BASE_SEP_26_2008_0446PM

#include <hpx/config.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/component_registry_base.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/capability.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a component_factory_base has to be used as a base class for all
    /// component factories.
    struct HPX_EXPORT component_factory_base
    {
        virtual ~component_factory_base() {}

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
        virtual component_type get_component_type(
            naming::gid_type const& locality, naming::resolver_client& agas_client) = 0;

        /// \brief Return the name of the component type this factory is
        ///        responsible for
        ///
        /// \return Returns the name of the component type this factory
        ///         instance is responsible for. This function throws on any
        ///         error.
        virtual std::string get_component_name() const = 0;

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
        virtual naming::gid_type create (std::size_t size = 1) = 0;

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function.
        ///
        /// \param f  [in] The constructor function to call in order to
        ///           initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance.
        virtual naming::gid_type create_with_args(
            util::function_nonser<void(void*)> const&) = 0;

        /// \brief Create one new component instance and initialize it using
        ///        the using the given constructor function. Assign the give
        ///        GID to the new object.
        ///
        /// \param assign_gid [in] The GID to assign to the newly created object.
        /// \param f  [in] The constructor function to call in order to
        ///           initialize the newly allocated object.
        ///
        /// \return   Returns the GID of the first newly created component
        ///           instance (this is the same as assign_gid, if successful).
        virtual naming::gid_type create_with_args(
            naming::gid_type const& assign_gid,
            util::function_nonser<void(void*)> const& f) = 0;

        /// \brief Destroy one or more component instances
        ///
        /// \param gid    [in] The gid of the first component instance to
        ///               destroy.
        /// \param addr   [in] The resolved address of the first component
        ///               instance to destroy.
        virtual void destroy(naming::gid_type const&,
            naming::address const& addr) = 0;

        /// \brief Ask whether this factory can be unloaded
        ///
        /// \return Returns whether it is safe to unload this factory and
        ///         the shared library implementing this factory. This
        ///         function will return 'true' whenever no more outstanding
        ///         instances of the managed object type are alive.
        bool may_unload() const
        {
            return instance_count() == 0;
        }

        /// \brief Ask how many instances are alive of the type this factory is
        ///        responsible for
        ///
        /// \return Returns the number of instances of the managed object type
        ///         which are currently alive.
        virtual long instance_count() const = 0;

#if defined(HPX_HAVE_SECURITY)
        /// \brief Return the required capabilities necessary to create an
        ///        instance of a component using this factory instance.
        ///
        /// \return Returns required capabilities necessary to create a new
        ///         instance of a component using this factory instance.
        virtual components::security::capability
            get_required_capabilities() const = 0;
#endif
    };
}}

namespace hpx { namespace util { namespace plugin
{
    ///////////////////////////////////////////////////////////////////////////
    // The following specialization of the virtual_constructor template
    // defines the argument list for the constructor of the concrete component
    // factory (derived from the component_factory_base above). This magic is needed
    // because we use hpx::plugin for the creation of instances of derived
    // types using the component_factory_base virtual base class only (essentially
    // implementing a virtual constructor).
    //
    // All derived component factories have to expose a constructor with the
    // matching signature. For instance:
    //
    //     class my_factory : public component_factory_base
    //     {
    //     public:
    //         my_factory (hpx::util::section const*, hpx::util::section const*, bool)
    //         {}
    //     };
    //
    template <>
    struct virtual_constructor<hpx::components::component_factory_base>
    {
        typedef
            hpx::util::detail::pack<
                hpx::util::section const*, hpx::util::section const*, bool
            > type;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_COMPONENT_FACTORY(FactoryType, componentname)            \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_COMPONENT_PREFIX,                            \
        hpx::components::component_factory_base, FactoryType,                 \
        componentname, factory);                                              \
    HPX_INIT_REGISTRY_FACTORY_STATIC(HPX_PLUGIN_COMPONENT_PREFIX,             \
        componentname, factory)                                               \
/**/

#define HPX_REGISTER_COMPONENT_FACTORY_DYNAMIC(FactoryType, componentname)    \
    HPX_PLUGIN_EXPORT_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX,                    \
        hpx::components::component_factory_base, FactoryType,                 \
        componentname, factory)                                               \
/**/

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_APPLICATION_NAME) && !defined(HPX_HAVE_STATIC_LINKING)
/// This macro is used to define the required Hpx.Plugin entry points. This
/// macro has to be used in exactly one compilation unit of a component module.
#define HPX_REGISTER_COMPONENT_MODULE()                                       \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)              \
    HPX_REGISTER_REGISTRY_MODULE()                                            \
/**/
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()                               \
    HPX_PLUGIN_EXPORT_LIST_DYNAMIC(HPX_PLUGIN_COMPONENT_PREFIX, factory)      \
    HPX_REGISTER_REGISTRY_MODULE_DYNAMIC()                                    \
/**/
#else
// in executables (when HPX_APPLICATION_NAME is defined) this needs to expand
// to nothing
#if defined(HPX_HAVE_STATIC_LINKING)
#define HPX_REGISTER_COMPONENT_MODULE()                                       \
    HPX_PLUGIN_EXPORT_LIST(HPX_PLUGIN_COMPONENT_PREFIX, factory)              \
    HPX_REGISTER_REGISTRY_MODULE()                                            \
/**/
#else
#define HPX_REGISTER_COMPONENT_MODULE()
#endif
#define HPX_REGISTER_COMPONENT_MODULE_DYNAMIC()
#endif

#endif

