//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_KERNEL2_AUG_14_2009_1030AM)
#define HPX_COMPONENTS_SERVER_KERNEL2_AUG_14_2009_1030AM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// The kernel2 is an HPX component.
    ///
    class HPX_COMPONENT_EXPORT kernel2
      : public simple_component_base<kernel2>
    {
    private:
        typedef simple_component_base<kernel2> base_type;
        
    public:
        kernel2();
        
        typedef hpx::components::server::kernel2 wrapping_type;
        
        enum actions
        {
            kernel2_large_set = 0,
            kernel2_large_set_local = 1
        };
        
        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        // This should go somewhere else ... but where?
        struct edge
        {
            edge()
            {}

            edge(naming::id_type const& source,
                 naming::id_type const& target,
                 int label)
              : source_(source), target_(target), label_(label)
            {}

            naming::id_type source_;
            naming::id_type target_;
            int label_;

        private:
            // serialization support
            friend class boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int)
            {
                ar & source_ & target_ & label_;
            }
        };

        typedef std::vector<edge> edge_list_type;

        typedef distributing_factory::locality_result locality_result;

        int
        large_set(naming::id_type G,
                  naming::id_type dist_edge_list);

        int
        large_set_local(locality_result local_list,
                        naming::id_type edge_list,
                        naming::id_type local_max_lco,
                        naming::id_type global_max_lco);

        typedef hpx::actions::result_action2<
            kernel2, int, kernel2_large_set,
            naming::id_type, naming::id_type,
            &kernel2::large_set
        > large_set_action;

        typedef hpx::actions::result_action4<
            kernel2, int, kernel2_large_set_local,
            locality_result, naming::id_type, naming::id_type, naming::id_type,
            &kernel2::large_set_local
        > large_set_local_action;
    };

}}}

#endif
