//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM)
#define HPX_COMPONENTS_DISTRIBUTING_FACTORY_JUN_20_2008_0948PM

#include <vector>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    // forward declaration
    class distributing_factory;

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    class distributing_factory
    {
    private:
        static component_type value;

    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component_base type used to encapsulate instances of this 
        // component
        typedef 
            managed_component_base<distributing_factory, server::distributing_factory> 
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
        static component_type get_component_type()
        {
            return value;
        }
        static void set_component_type(component_type type)
        {
            value = type;
        }

        // constructor
        distributing_factory()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        struct locality_result
        {
            locality_result()
            {}

            locality_result(naming::id_type const& prefix, 
                    naming::id_type const& first_gid, std::size_t count)
              : prefix_(prefix), first_gid_(first_gid), count_(count)
            {}

            naming::id_type prefix_;    ///< prefix of the locality 
            naming::id_type first_gid_; ///< gid of the first created component
            std::size_t count_;         ///< nu,ber of created components

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & prefix_ & first_gid_ & count_;
            }
        };

        typedef std::vector<locality_result> result_type;

        /// \brief Action to create new components
        threads::thread_state create_components(
            threads::thread_self& self, applier::applier& app, 
            result_type* gids, components::component_type type, 
            std::size_t count); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory, result_type, factory_create_component, 
            components::component_type, std::size_t, 
            &distributing_factory::create_components
        > create_components_action;
    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    class distributing_factory 
      : public managed_component_base<
            detail::distributing_factory, distributing_factory
        >
    {
    public:
        typedef detail::distributing_factory wrapped_type;
        typedef managed_component_base<wrapped_type, distributing_factory> base_type;

        distributing_factory(applier::applier&)
          : base_type(new wrapped_type())
        {}
    };

}}}

#endif
