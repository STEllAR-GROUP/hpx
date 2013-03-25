//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PLUGIN_FACTORY_BASE_MAR_24_2013_0333PM)
#define HPX_PLUGIN_FACTORY_BASE_MAR_24_2013_0333PM

#include <hpx/config.hpp>
#include <hpx/plugins/plugin_registry_base.hpp>

#include <hpx/util/plugin.hpp>
#include <hpx/util/plugin/export_plugin.hpp>

#include <boost/mpl/list.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a plugin_factory_base has to be used as a base class for all
    /// plugin factories.
    struct HPX_EXPORT plugin_factory_base
    {
        virtual ~plugin_factory_base() {}
    };
}}

namespace hpx { namespace util { namespace plugin
{
    ///////////////////////////////////////////////////////////////////////////
    // The following specialization of the virtual_constructors template
    // defines the argument list for the constructor of the concrete component
    // factory (derived from the component_factory_base above). This magic is needed
    // because we use hpx::plugin for the creation of instances of derived
    // types using the component_factory_base virtual base class only (essentially
    // implementing a virtual constructor).
    //
    // All derived component factories have to expose a constructor with the
    // matching signature. For instance:
    //
    //     class my_factory : public plugin_factory_base
    //     {
    //     public:
    //         my_factory (hpx::util::section const*, hpx::util::section const*, bool)
    //         {}
    //     };
    //
    template <>
    struct virtual_constructors<hpx::plugins::plugin_factory_base>
    {
        typedef boost::mpl::list<
            boost::mpl::list<
                hpx::util::section const*, hpx::util::section const*, bool
            >
        > type;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
/// This macro is used to register the given component factory with
/// Hpx.Plugin. This macro has to be used for each of the component factories.
#define HPX_REGISTER_PLUGIN_FACTORY_BASE(FactoryType, componentname)          \
    HPX_PLUGIN_EXPORT(HPX_PLUGIN_PREFIX,                                      \
        hpx::plugins::plugin_factory_base, FactoryType,                       \
        pluginname, factory)                                                  \
/**/

#endif

