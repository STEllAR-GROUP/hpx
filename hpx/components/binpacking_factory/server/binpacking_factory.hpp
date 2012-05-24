//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/locality_result.hpp>

#include <vector>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT binpacking_factory
      : public simple_component_base<binpacking_factory>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object
        enum actions
        {
            factory_create_components = 0,
                // create new components, based on current population
            factory_create_components_counterbased = 1
                // create new components, based on performance counter
        };

        // constructor
        binpacking_factory()
        {}

        typedef std::vector<util::remote_locality_result> remote_result_type;
        typedef std::vector<util::locality_result> result_type;

        typedef util::locality_result_iterator iterator_type;
        typedef
            std::pair<util::locality_result_iterator, util::locality_result_iterator>
        iterator_range_type;

        /// \brief Action to create new components
        remote_result_type create_components(components::component_type type,
            std::size_t count) const;

        /// \brief Action to create new components based on given performance
        ///        counter
        remote_result_type create_components_counterbased(
            components::component_type type, std::size_t count,
            std::string const&) const;

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            binpacking_factory const, remote_result_type,
            factory_create_components, components::component_type, std::size_t,
            &binpacking_factory::create_components
        > create_components_action;

        typedef hpx::actions::result_action3<
            binpacking_factory const, remote_result_type,
            factory_create_components_counterbased,
            components::component_type, std::size_t, std::string const&,
            &binpacking_factory::create_components_counterbased
        > create_components_counterbased_action;
    };
}}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the distributing_factory actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::binpacking_factory::create_components_action
  , distributing_factory_create_components_action
)

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::components::server::binpacking_factory::create_components_counterbased_action
  , distributing_factory_create_components_counterbased_action
)

#endif
