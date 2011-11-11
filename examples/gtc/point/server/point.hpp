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

#include "../../particle/server/particle.hpp"

#include <iostream>
#include <fstream>

// this is just to optimize performance a little -- this number can be exceeded
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
            point_search = 1
        };

        // constructor: initialize accumulator value
        point()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(std::size_t objectid,std::string meshfile)
        {
            idx_ = objectid;
            neighbors_.reserve(MAX_NUM_NEIGHBORS);

            std::string line;
            std::string val1,val2,val3,val4;
            std::ifstream myfile;
            myfile.open(meshfile);
            if (myfile.is_open() ) {
              while (myfile.good()) { 
                while(std::getline(myfile,line)) {
                  std::istringstream isstream(line);
                  std::getline(isstream,val1,' ');
                  std::getline(isstream,val2,' ');
                  std::getline(isstream,val3,' ');
                  std::getline(isstream,val4,' ');
                  std::size_t node = atoi(val1.c_str());   
                  double posx = atof(val2.c_str());   
                  double posy = atof(val3.c_str());   
                  double posz = atof(val4.c_str());   
                  if ( node == objectid ) {
                    posx_ = posx;
                    posy_ = posy;
                    posz_ = posz;
                  }
                }
              }
              myfile.close(); 
            } 
           
        }

        int search(std::vector<hpx::naming::id_type> const& particle_components);
        
        bool search_callback(std::size_t i, double const& distance);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action2<
            point, point_init,std::size_t,std::string, &point::init
        > init_action;

        typedef hpx::actions::result_action1<
            point, int,point_search, std::vector<hpx::naming::id_type> const&, &point::search
        > search_action;

    private:
        //hpx::lcos::local_mutex mtx_;    // lock for this data block

        std::size_t idx_;
        std::vector<std::size_t> neighbors_;
        double posx_,posy_,posz_;
    };

}}}

#endif
