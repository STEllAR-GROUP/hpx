//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/component_type.hpp>
#include <hpx/runtime/actions/action.hpp>

namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class distributing_factory
    {
    public:
        // components must contain a typedef for wrapping_type defining the
        // wrapper type used to encapsulate instances of this component
        typedef 
            wrapper<distributing_factory, server::distributing_factory> 
        wrapping_type;

        // parcel action code: the action to be performed on the destination 
        // object 
        enum actions
        {
            factory_create_component = 0,  // create new components
        };

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = component_distributing_factory };

        // constructor
        distributing_factory()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// \brief Action to create new components
        threads::thread_state create(
            threads::thread_self& self, applier::applier& app,
            naming::id_type* gid, components::component_type type, 
            std::size_t count); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory, naming::id_type, factory_create_component, 
            components::component_type, std::size_t, 
            &distributing_factory::create
        > create_action;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    class distributing_factory 
      : public wrapper<detail::distributing_factory, distributing_factory>
    {
    private:
        typedef detail::distributing_factory wrapped_type;
        typedef wrapper<wrapped_type, distributing_factory> base_type;

    public:
        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        enum { value = wrapped_type::value };

        distributing_factory()
          : base_type(new wrapped_type())
        {}

    protected:
        base_type& base() { return *this; }
        base_type const& base() const { return *this; }
    };

}}}

///////////////////////////////////////////////////////////////////////////////
// Serialization support for the accumulator actions
HPX_SERIALIZE_ACTION(hpx::components::server::detail::distributing_factory::create_action);

#endif
