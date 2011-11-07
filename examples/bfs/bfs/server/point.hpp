//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_POINT)
#define HPX_COMPONENTS_SERVER_POINT

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/local_mutex.hpp>

#include <iostream>
#include <fstream>

#define MAX_NUM_NEIGHBORS 20

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT point
      : public components::detail::managed_component_base<point>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            point_init = 0,
            point_traverse = 2
        };

        // constructor: initialize accumulator value
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(std::size_t objectid,std::string graphfile)
        {
            idx_ = objectid;
            visited_ = false;
            neighbors_.reserve(MAX_NUM_NEIGHBORS);

            std::string line;
            std::string val1,val2;
            std::ifstream myfile;
            myfile.open(graphfile);
            if (myfile.is_open() ) {
              while (myfile.good()) { 
                while(std::getline(myfile,line)) {
                  std::istringstream isstream(line);
                  std::getline(isstream,val1,' ');
                  std::getline(isstream,val2,' ');
                  std::size_t node = atoi(val1.c_str());   
                  std::size_t neighbor = atoi(val2.c_str());   
                  if ( node == objectid ) {
                    neighbors_.push_back(neighbor); 
                  }
                }
              }
            } 
        }

        // traverse the tree
        std::vector<std::size_t> traverse(std::size_t level, std::size_t parent);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action2<
            point, point_init,std::size_t,std::string, &point::init
        > init_action;

        typedef hpx::actions::result_action2<
            point, std::vector<std::size_t>,point_traverse, std::size_t, std::size_t, &point::traverse
        > traverse_action;

    private:
        //hpx::lcos::local_mutex mtx_;    // lock for this data block

        std::size_t idx_;
        std::size_t level_;
        bool visited_;
        std::vector<std::size_t> neighbors_;
        std::size_t parent_;
    };

}}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::geometry::server::point::init_action,
    bfs_geometry_init_action);

HPX_REGISTER_ACTION_DECLARATION_EX(
    hpx::geometry::server::point::traverse_action,
    bfs_geometry_traverse_action);

#endif
