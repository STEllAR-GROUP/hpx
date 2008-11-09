//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_FACTORY_BASE_SEP_26_2008_0446PM)
#define HPX_COMPONENT_FACTORY_BASE_SEP_26_2008_0446PM

#include <boost/plugin.hpp>
#include <boost/plugin/export_plugin.hpp>
#include <boost/mpl/list.hpp>

#include <hpx/config.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_type.hpp>

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_COMPONENT_NAME)
# define HPX_COMPONENT_NAME hpx
#endif

///////////////////////////////////////////////////////////////////////////////
#if !defined(HPX_COMPONENT_LIB_NAME)
#define HPX_COMPONENT_LIB_NAME                                                \
        HPX_MANGLE_COMPONENT_NAME(HPX_COMPONENT_NAME)                         \
    /**/
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class component_factory_base component_factory_base.hpp hpx/runtime/components/component_factory_base.hpp
    ///
    /// The \a component_factory_base has to be used as a base class for all 
    /// component factories.
    struct HPX_EXPORT component_factory_base
    {
        virtual ~component_factory_base() {}

        /// \brief Return the unique identifier of the component type this 
        ///        factory is responsible for
        ///
        /// \param prefix       [in] The prefix of the locality this factory 
        ///                     is responsible for.
        /// \param agas_client  [in] The AGAS client to use for component id 
        ///                     registration (if needed).
        ///
        /// \return Returns the unique identifier of the component type this 
        ///         factory instance is responsible for. This function throws
        ///         on any error.
        virtual component_type get_component_type(
            naming::id_type const& prefix, naming::resolver_client& agas_client) = 0;

        /// \brief Return the name of the component type this factory is 
        ///        responsible for
        ///
        /// \return Returns the name of the component type this factory 
        ///         instance is responsible for. This function throws on any 
        ///         error.
        virtual std::string get_component_name() const = 0;

        /// \brief  The function \a get_factory_properties is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        virtual factory_property get_factory_properties() const = 0;

        /// \brief Create one or more new component instances.
        ///
        /// \param self   [in] The PX \a thread used to execute this function.
        /// \param appl   [in] The applier instance to be used to create
        ///               the new component instances.
        /// \param count  [in] The number of component instances to 
        ///               create. The value of this parameter should not 
        ///               be zero.
        ///
        /// \return Returns the GID of the first newly created component 
        ///         instance. If more than one component instance has been 
        ///         created (\a count > 1) the GID's of all new instances are
        ///         sequential in a row.
        virtual naming::id_type create (threads::thread_self& self, 
            applier::applier& appl, std::size_t count) = 0;

        /// \brief Destroy one or more component instances
        ///
        /// \param self   [in] The PX \a thread used to execute this function.
        /// \param appl   [in] The applier instance to be used to destroy
        ///               the component instances.
        /// \param gid    [in] The gid of the first component instance to 
        ///               destroy. 
        virtual void destroy(threads::thread_self& self, 
            applier::applier& appl, naming::id_type const& gid) = 0;
    };

}}

namespace boost { namespace plugin 
{
    ///////////////////////////////////////////////////////////////////////////
    // The following specialization of the virtual_constructors template 
    // defines the argument list for the constructor of the concrete component
    // factory (derived from the component_factory_base above). This magic is needed 
    // because we use boost::plugin for the creation of instances of derived 
    // types using the component_factory_base virtual base class only (essentially
    // implementing a virtual constructor).
    //
    // All derived component factories have to expose a constructor with the
    // matching signature. For instance:
    //
    //     class my_factory : public component_factory_base
    //     {
    //     public:
    //         my_factory (hpx::util::section const*, hpx::util::section const*)
    //         {}
    //     };
    //
    template<>
    struct virtual_constructors<hpx::components::component_factory_base>
    {
        typedef mpl::list<
            mpl::list<hpx::util::section const*, hpx::util::section const*>
        > type;
    };

}}

///////////////////////////////////////////////////////////////////////////////
/// The macro \a HPX_REGISTER_COMPONENT_FACTORY is used to register the given
/// component factory with Boost.Plugin. This macro has to be used for each of 
/// the component factories.
#define HPX_REGISTER_COMPONENT_FACTORY(FactoryType, componentname)            \
        BOOST_PLUGIN_EXPORT(HPX_COMPONENT_LIB_NAME,                           \
            hpx::components::component_factory_base, FactoryType,             \
            componentname, factory)                                           \
    /**/

/// The macro \a HPX_REGISTER_COMPONENT_MODULE is used to define the required
/// Boost.Plugin entry points. This macro has to be used in exactly one 
/// compilation unit of a component module.
#define HPX_REGISTER_COMPONENT_MODULE()                                       \
        BOOST_PLUGIN_EXPORT_LIST(HPX_COMPONENT_LIB_NAME, factory);            \
    /**/

#endif
