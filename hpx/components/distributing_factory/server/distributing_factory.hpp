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

#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT distributing_factory
      : public simple_component_base<distributing_factory>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object 
        enum actions
        {
            factory_create_components = 0,  // create new components
            factory_free_components = 1,    // free existing components
        };

        // constructor
        distributing_factory(applier::applier& appl)
          : simple_component_base<distributing_factory>(appl)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        struct locality_result
        {
            locality_result()
            {}

            locality_result(naming::id_type const& prefix, 
                    naming::id_type const& first_gid, std::size_t count,
                    components::component_type type)
              : prefix_(prefix), first_gid_(first_gid), count_(count), 
                type_(type)
            {}

            naming::id_type prefix_;    ///< prefix of the locality 
            naming::id_type first_gid_; ///< gid of the first created component
            std::size_t count_;         ///< number of created components
            components::component_type type_; ///< type of created components

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

        /// \brief Action to delete existing components
        threads::thread_state free_components(
            threads::thread_self& self, applier::applier& app,
            result_type const& gids); 

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into a action
        // type, allowing to generate all require boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action2<
            distributing_factory, result_type, factory_create_components, 
            components::component_type, std::size_t, 
            &distributing_factory::create_components
        > create_components_action;

        typedef hpx::actions::action1<
            distributing_factory, factory_free_components, 
            result_type const&, &distributing_factory::free_components
        > free_components_action;
    };

}}}

#endif
