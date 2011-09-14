///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2011 Daniel Kogler
//
//  Distributed under the Boost Software License, Version 1.0.(See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////
#ifndef _BHCONTROLLER_SERVER_HPP
#define _BHCONTROLLER_SERVER_HPP

/*This is the bhcontroller class implementation header file.
*/

#include <time.h>

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include <boost/foreach.hpp>
#include <boost/random/linear_congruential.hpp>
#include "../bhnode.hpp"
#include "bhnode.cpp"

#include <boost/date_time/posix_time/posix_time.hpp>

using namespace boost::posix_time;
using hpx::applier::get_applier;

namespace hpx { namespace components { namespace server
{
    class HPX_COMPONENT_EXPORT bhcontroller : 
          public simple_component_base<bhcontroller>
    {
    public:
    //enumerate all of the actions that will(or can) be employed
    enum actions{
        hpl_construct,
        hpl_run
    };

    //constructors and destructor
    bhcontroller(){}
    int construct(id_type gid, std::string input);
    int run_simulation();
    ~bhcontroller(){destruct();}
    void destruct();

    private:
    void build_init();

    //data members
    id_type _gid;
    int numBodies;
    int numIters;
    double dtime, eps, tolerance;
    double halfdt, softening, invTolerance;
    double** particlePositions;
    double*  particleMasses;
    components::bhnode treeRoot;
    components::bhnode* particle;
    vector<components::bhnode> regions;

    public:
    //here we define the actions that will be used
    typedef hpx::actions::result_action2<bhcontroller, int, hpl_construct,
        id_type, std::string, &bhcontroller::construct> constructAction;
    typedef hpx::actions::result_action0<bhcontroller, int, hpl_run,
        &bhcontroller::run_simulation> runAction;

    //here begins the definitions of most of the future types that will be used

    };
///////////////////////////////////////////////////////////////////////////////

    //the constructor initializes the matrix
    int bhcontroller::construct(naming::id_type gid, std::string inputFile){
    std::vector<bhnode::constFuture> futures;
    std::ifstream infile;
    infile.open(inputFile.c_str());
    if(!infile){
        std::cerr<<"Can't open input file "<<inputFile<<std::endl;
        return 1;
    }
    try{
        double dat[7];
        double bounds[6] = {-1.7e308,-1.7e308,-1.7e308,1.7e308,1.7e308,1.7e308};

        infile>>numBodies>>numIters>>dtime>>eps>>tolerance;
        particle = new components::bhnode[numBodies];
        particlePositions = new double*[numBodies];
        particleMasses = new double[numBodies];
        for(int i = 0; i < numBodies; i++){
            particlePositions[i] = new double[3];
            particle[i].create(get_applier().get_runtime_support_gid());
            infile>>dat[0]>>dat[1]>>dat[2]>>dat[3]>>dat[4]>>dat[5]>>dat[6];
            futures.push_back(particle[i].construct_node(dat,false));
            for(int j = 0; j < 3; j++){
                if(bounds[j] < dat[j+1]){bounds[j] = dat[j+1];}
                if(bounds[j+3] > dat[j+1]){bounds[j+3] = dat[j+1];}
                particlePositions[i][j] = dat[j+1];
            }
            particleMasses[i] = dat[0];
        }

        treeRoot.create(get_applier().get_runtime_support_gid());
        id_type treeGid = treeRoot.get_gid();
        dat[0] = 0;
        for(int i = 0; i < 3; i++){
            dat[i+1] = (bounds[i]+bounds[i+3])*.5;
            dat[i+4] = 0;
        }
        futures.push_back(treeRoot.construct_node(dat,true));
        halfdt = .5 * dtime;
        softening = eps * eps;
        invTolerance = 1 / (tolerance * tolerance);
        treeRoot.set_boundaries(treeGid, bounds);
        regions.push_back(treeGid);
        for(int i = 0; i < numBodies+1; i++){futures[i].get();}
    }
    catch(...){
        std::cerr<<"An error occurred while reading the file "<<inputFile<<"\n";
        infile.close();
        return 1;
    }
    infile.close();
    _gid = gid;
    return 0;
    }

    //the destructor frees the memory
    void bhcontroller::destruct(){
        treeRoot.free();
    }

    int bhcontroller::run_simulation(){
ptime start = microsec_clock::local_time();
        build_init();
std::cout<<microsec_clock::local_time() - start<<std::endl;
        vector<bhnode::runFuture> simFutures;
        vector<double> runParams;
        {
            bhnode::printFuture future(treeRoot.get_gid(), 0, 8);
            future.get();
        }
        runParams.push_back(numIters);
        runParams.push_back(dtime);
        runParams.push_back(eps);
        runParams.push_back(tolerance);
        for(int i = 0; i < numBodies; i++){
            simFutures.push_back(particle[i].run(_gid, runParams));
        }
        for(int i = 0; i < numBodies; i++){simFutures[i].get();}
        return 0;
    }

    void bhcontroller::build_init(){
        vector<bhnode::inPntFuture> futures;
        for(int i = 0; i < numBodies; i++){
            vector<double> pos(particlePositions[i],
                               particlePositions[i]+3);
            futures.push_back(treeRoot.insert_node(pos, particleMasses[i],
                                        particle[i].get_gid()));
        }
        for(int i = 0; i < numBodies; i++){futures[i].get();}
    }

}}}

#endif
