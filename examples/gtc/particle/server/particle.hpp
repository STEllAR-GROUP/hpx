//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_PARTICLE)
#define HPX_COMPONENTS_SERVER_PARTICLE

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/lcos/local_mutex.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace geometry { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT particle
      : public components::detail::managed_component_base<particle>
    {
    public:
        // parcel action code: the action to be performed on the destination
        // object (the accumulator)
        enum actions
        {
            particle_init = 0,
            particle_distance = 1
        };

        // constructor: initialize accumulator value
        particle()
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        void init(std::size_t objectid,std::string particlefile)
        {
            idx_ = objectid;

            std::string line;
            std::string val1,val2,val3,val4;
            std::ifstream myfile;
            myfile.open(particlefile);
            if (myfile.is_open() ) {
              while (myfile.good()) { 
                while(std::getline(myfile,line)) {
                  std::istringstream isstream(line);
                  std::getline(isstream,val1,' ');
                  std::getline(isstream,val2,' ');
                  std::getline(isstream,val3,' ');
                  std::getline(isstream,val4,' ');
                  std::size_t node = atoi(val1.c_str());   
                  double posx = atoi(val2.c_str());   
                  double posy = atoi(val3.c_str());   
                  double posz = atoi(val4.c_str());   
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

        /// Initialize the accumulator
        double distance(double posx,double posy,double posz)
        {
          return sqrt( pow(posx_ - posx,2) + pow(posy_ - posy,2) + pow(posz_ - posz,2) );
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action2<
            particle, particle_init,std::size_t,std::string, &particle::init
        > init_action;

        typedef hpx::actions::result_action3<
            particle, double, particle_distance, double,double,double,
            &particle::distance
        > distance_action;

    private:
        hpx::lcos::local_mutex mtx_;    // lock for this data block
        std::size_t idx_;
        double posx_,posy_,posz_;
    };

}}}

#endif
