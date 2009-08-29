//  Copyright (c) 2007-2009 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_EDGE_AUG_28_2009_0447PM)
#define HPX_COMPONENTS_SERVER_EDGE_AUG_28_2009_0447PM

#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The edge is an HPX component.
    ///
    class edge
      : public components::detail::managed_component_base<edge>
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the edge)
        enum actions
        {
            edge_init = 0,
            edge_get_snapshot = 1
        };
        
        // constructor: initialize edge value
        edge()
          : label_(-1)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        struct edge_snapshot
        {
            edge_snapshot()
              : label_(-1)
            {}

            edge_snapshot(naming::id_type source, naming::id_type target, int label)
              : source_(source), target_(target), label_(label)
            {}

            naming::id_type source_, target_;
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
        typedef edge_snapshot edge_snapshot_type;

        //typedef hpx::components::edge::edge_snapshot edge_snapshot_type;
        //typedef hpx::components::edge::edge_snapshot_type edge_snapshot_type;

        /// Initialize the edge
        // This is an opt. for when we know the order a priori
        int init(naming::id_type source, naming::id_type target, int label)
        {
            source_ = source;
            target_ = target;
            label_ = label;

            return 0;
        }

        edge_snapshot_type get_snapshot(void)
        {
            return edge_snapshot_type(source_, target_, label_);
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::result_action3<
            edge, int, edge_init, naming::id_type, naming::id_type, int , &edge::init
        > init_action;

        typedef hpx::actions::result_action0<
            edge, edge_snapshot_type, edge_get_snapshot, &edge::get_snapshot
        > get_snapshot_action;

    private:
        naming::id_type source_;
        naming::id_type target_;
        int label_;
    };

}}}

#endif
