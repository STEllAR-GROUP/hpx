// Copyright (c) 2011 John West <john.e.west@gmail.com>
// Copyright (c) 2011 Matt Anderson <matt@phys.lsu.edu>
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This example code is a candidate for the "next simplest possible" HPX program that is in some way
// instructive. As such the code in it may not represent the best way of getting something done;
// it is primarily meant to familiarize the reader with the essential steps to getting an HPX 
// code up and running.
//
// The code starts out in the main routine, but the only function of the main() in this example is to 
// (using the Boost helper routines) package up the command line options and call hpx_init(). 
// hpx_init() initializes the runtime environment and then calls hpx_main(). This routine must be present, 
// by name, in any HPX program. (For more about Boost, see http://www.boost.org/ .) 
//
// hpx_main() does some bookkeeping and setup and then causes parallel work to be done by invoking
// the futures registered earlier in the program.
// 
// To run the program
//       ./basic_example -r -t 2 -a localhost:5005 -x localhost:5006. 
//       - The "-t 2" option runs this program on two OS threads (one for each future). 
//       - The "-r" option runs the AGAS server (necessary if it isn't already running somewhere else).
//       - The "-a localhost:5005" option places the AGAS server on your machine and specifies that it
//         use port 5005 (this example assumes that ports 5005 and 5006 are open on your system. 
//         If they aren't, substitute two open port numbers appropriate for you).
//       - The "-x localhost:5006" specifies that HPX should include "localhost" in the runtime and use
//         port 5006.
// 
// If you'd like to get some validation that getnumber_action is being run on two different threads, you 
// can ask HPX to output a logfile via an environment variable. If you are using bash type
// 
//    export HPX_LOGLEVEL=6
// 
// for very detailed information or
// 
//    export HPX_LOGLEVEL=2
// 
// for more high level information.
// 
// After you execute the application HPX creates a log file of the form hpx.PID.log. You can search this file
// for the string getnumber_action and see that the thread IDs differ.

//C++ include to permit output to the screen
#include <stdio.h>
#include <iostream>
#include <algorithm> // for reverse, unique
#include <string>

//HPX includes
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

//Boost includes
#include <math.h>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>
#include "boost/geometry/geometry.hpp"
#include <boost/geometry/geometries/geometries.hpp>
#include "boost/geometry/geometries/cartesian2d.hpp"
#include <boost/geometry/geometries/adapted/c_array_cartesian.hpp>
#include <boost/geometry/geometries/adapted/std_as_linestring.hpp>
#include <boost/geometry/multi/multi.hpp>

#include "boost/geometry/extensions/index/rtree/rtree.hpp"

using namespace hpx;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description desc_commandline ("usage:basic_example");

    int retcode = hpx::init(desc_commandline, argc, argv);
    return retcode;

}

naming::id_type get_initialdata(int);
int search_iterate(naming::id_type,naming::id_type,naming::id_type,naming::id_type,int);

typedef 
    actions::plain_result_action1<naming::id_type,int, get_initialdata> 
get_initialdata_action;

typedef 
    actions::plain_result_action5<int,naming::id_type,naming::id_type,naming::id_type,naming::id_type,int, search_iterate> 
search_iterate_action;

HPX_REGISTER_PLAIN_ACTION(get_initialdata_action);
HPX_REGISTER_PLAIN_ACTION(search_iterate_action);

struct data
{
    data()
    {}
    ~data() {}

    std::vector<double> slaves1;
    std::vector<double> slaves2;
    std::vector<double> vel1;
    std::vector<double> vel2;
    boost::geometry::polygon_2d element;

private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & slaves1 & slaves2 & vel1 & vel2;
    }
};

hpx::actions::manage_object_action<data> const manage_data =
        hpx::actions::manage_object_action<data>();

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::actions::manage_object_action<data>, manage_object_action_data)


int hpx_main(po::variables_map &vm)
{
    double elapsed = 0.0;

    std::vector<naming::id_type> prefixes;
    applier::applier& appl = applier::get_applier();

    naming::id_type this_prefix = appl.get_runtime_support_gid();
    
    components::component_type type = 
        components::get_component_type<components::server::plain_function<get_initialdata_action> >();
    naming::id_type that_prefix;
    
    if (appl.get_remote_prefixes(prefixes, type)) {
      that_prefix = prefixes[0];
    } else {
      that_prefix = this_prefix;
    }

    
    {
        // Create a timer so see how its done
        util::high_resolution_timer t;

        // Create the first object
        lcos::eager_future<get_initialdata_action> n1(that_prefix,0);
        lcos::eager_future<get_initialdata_action> n2(this_prefix,1);

        // duplicate initial state (for concurrent search/contact iteration)
        lcos::eager_future<get_initialdata_action> contact_n1(that_prefix,0);
        lcos::eager_future<get_initialdata_action> contact_n2(this_prefix,1);

        //components::access_memory_block<data> val( components::stubs::memory_block::get(n1.get()) );
        //for (int i=0;i<val->slaves.size();i++) {
        //  if ( i%2 == 0 ) lcos::eager_future<search_iterate_action> s1(that_prefix,n1.get(),n2.get(),n3.get(),n4.get(),i);
        //  else lcos::eager_future<search_iterate_action> s1(that_prefix,n1.get(),n2.get(),n3.get(),n4.get(),i);
        //}
        lcos::eager_future<search_iterate_action> s1(that_prefix,n1.get(),n2.get(),contact_n1.get(),contact_n2.get(),0); 
        lcos::eager_future<search_iterate_action> s2(this_prefix,n1.get(),n2.get(),contact_n1.get(),contact_n2.get(),1); 
        lcos::eager_future<search_iterate_action> s3(that_prefix,n1.get(),n2.get(),contact_n1.get(),contact_n2.get(),2); 
        lcos::eager_future<search_iterate_action> s4(this_prefix,n1.get(),n2.get(),contact_n1.get(),contact_n2.get(),3); 
         
        // Get the final velocities
        //components::access_memory_block<data> val1( components::stubs::memory_block::get(contact_n1.get()) );
        //components::access_memory_block<data> val2( components::stubs::memory_block::get(contact_n2.get()) );

        // What is the elapsed time?
        elapsed = t.elapsed();

        // Print out a completion message. 
        //for (int i=0;i<val1->vel1.size();i++) {
        //  std::cout << "Final velocities element A " << val1->vel1[i] << " " << val1->vel2[i] << std::endl;
        //}
        //for (int i=0;i<val2->vel1.size();i++) {
        //  std::cout << "Final velocities element B " << val2->vel1[i] << " " << val2->vel2[i] << std::endl;
        //}
        std::cout << "Elapsed " << elapsed << " seconds."<< std::endl;
    }

    // Initiate shutdown of the runtime systems on all localities
    hpx::finalize();
    return 0;
}

naming::id_type get_initialdata(int index)
{  
    naming::id_type here = applier::get_applier().get_runtime_support_gid();    
    naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(data), manage_data); 
    components::access_memory_block<data> val(
                components::stubs::memory_block::checkout(result));
    
    int locality = get_prefix_from_id( here );

    double h = 1.0;
    double V_0 = 1.0;
    double dt = 0.1;

    // We start the contact search algorithm two timesteps after this initial position
    double coorA[][2] = {
        {h/4.0, h}, {h/4.0-h/sqrt(2.0),h+h/sqrt(2.0)}, {h/4,h+sqrt(2.0)*h}, {h/4.0+h/sqrt(2.0),h+h/sqrt(2.0)}
        };
    double velA[][2] = {
        {0.0,-V_0}, {0.0,-V_0}, {0.0,-V_0}, {0.0,-V_0}
        };

    double coorB[][2] = {
        {0.0,0.0}, {0.0,h}, {h,h}, {h,0.0}
        };
    double velB[][2] = {
        {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0}
        };

    for (int i=0;i<4;i++) {
      for (int j=0;j<2;j++) {
        coorA[i][j] += dt*velA[i][j];
        coorB[i][j] += dt*velB[i][j];
      }
    }


    if ( index == 0 ) {
      boost::geometry::assign(val->element,coorA);
      correct(val->element);
      for (int i=0;i<4;i++) {
        val->slaves1.push_back(coorA[i][0]);
        val->slaves2.push_back(coorA[i][1]);
        val->vel1.push_back(velA[i][0]);
        val->vel2.push_back(velA[i][1]);
      }
    } else {
      boost::geometry::assign(val->element,coorB);
      correct(val->element);
      for (int i=0;i<4;i++) {
        val->slaves1.push_back(coorB[i][0]);
        val->slaves2.push_back(coorB[i][1]);
        val->vel1.push_back(velB[i][0]);
        val->vel2.push_back(velB[i][1]);
      }
    }

    return result;
}

int search_iterate(naming::id_type n1,naming::id_type n2,naming::id_type cn1, naming::id_type cn2,int index)
{
  components::access_memory_block<data> val1(
                components::stubs::memory_block::checkout(n1));

  components::access_memory_block<data> val2(
                components::stubs::memory_block::checkout(n2));

  components::access_memory_block<data> val3(
                components::stubs::memory_block::checkout(cn1));

  components::access_memory_block<data> val4(
                components::stubs::memory_block::checkout(cn2));

  if ( boost::geometry::within(boost::geometry::make<boost::geometry::point_2d>
                                 (val1->slaves1[index],val1->slaves2[index]), val2->element) ) {
    // Contact!
    // Compute data necessary for contact iteration 
    int master_segment = 1;
    int N_1 = master_segment;
    int N_2 = master_segment + 1;
    
    boost::geometry::point_2d p1(val2->slaves1[N_1],val2->slaves2[N_1]); 
    boost::geometry::point_2d p2(val2->slaves1[N_2],val2->slaves2[N_2]); 
    double l = boost::geometry::distance(p1,p2);
    
    // Eqn. 2
    double A = (val2->slaves2[N_2] - val2->slaves2[N_1])/l;
    double B = (val2->slaves1[N_1] - val2->slaves1[N_2])/l;
    double C = (val2->slaves1[N_2]*val2->slaves2[N_1] - val2->slaves1[N_1]*val2->slaves2[N_2])/l;
    double delta = -(A*val1->slaves1[index] + B*val1->slaves2[index] + C);

    // Eqn. 6-7
    double xsm = val1->slaves1[index] + A*delta;
    double zsm = val1->slaves2[index] + B*delta;

    boost::geometry::point_2d p3(xsm,zsm);
    double R_1 = distance(p2,p3)/l;
    double R_2 = 1.0 - R_1;

    // begin contact iteration
    int N = 1;  // number of contact iterations
    for (int n=0;n<N;n++) { 
      // Eqn. 12 -- modified since Gordon Brown is thinking Fortran indexing instead of C++
      double alpha1 = 1./sqrt(N-(n+1)+1);

      double alpha2 = 1; // the slave node N_s is never a master node in this example

      // match the normal velocity of the slave node to the normal velocity of the master segment
      // Eqn. 11
      double alpha = alpha1*alpha2;

      // Eqn. 10
      double dt = 0.1;
      double dv = -alpha*delta/dt/(1. + pow(R_1,2) + pow(R_2,2));

      // update the velocities of the slave node (no friction in this example, else use Eqn. 16)
      val3->vel1[index] += -A*dv;   
      val3->vel2[index] += -B*dv;   
      std::cout << " Change for A : " << -A*dv << " " << -B*dv << std::endl;

      // update the velocities of the master segment nodes
      val4->vel1[N_1] = -R_1*dv;
      val4->vel1[N_2] = -R_2*dv;
      std::cout << " Change for B : " << -R_1*dv << " " << -R_2*dv << std::endl;
    }
  }

  return 0;
}
