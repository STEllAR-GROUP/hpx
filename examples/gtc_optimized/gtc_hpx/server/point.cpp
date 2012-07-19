//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future_wait.hpp>

#include "../../fname.h"
#include "point.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

extern "C" {void FNAME(setup)(int *,int *,int *,int *,int *, int *); 
            void FNAME(load)(); 
            void FNAME(chargei)(void* opaque_ptr_to_class); 
            void FNAME(partd_allreduce_cmm) (void* pfoo) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = static_cast<gtc::server::point*>(pfoo); 
                    ptr_to_class->partd_allreduce();
                    return; };
}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::setup(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;
      generation_ = 0;

      // TEST
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      npartdom = 5;
      ntoroidal = 2;
      int particle_domain_location=mype%npartdom;
      int toroidal_domain_location=mype/npartdom;
      int myrank_toroidal = toroidal_domain_location;
      if ( myrank_toroidal > particle_domain_location ) {
        myrank_toroidal = particle_domain_location;
      }
      hpx_left_pe = (myrank_toroidal-1+ntoroidal)%ntoroidal;
      hpx_right_pe = (myrank_toroidal+1)%ntoroidal;
#if 0
      int t1 = numberpe;
      int t2 = mype;
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      FNAME(setup)(&t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);

      FNAME(load)();
#endif

      // Figure out the communicators: toroidal_comm and partd_comm
      int my_pdl = mype%npartdom;
      int my_tdl = mype/npartdom;
      for (std::size_t i=0;i<numberpe;i++) {
        int particle_domain_location= i%npartdom;
        int toroidal_domain_location= i/npartdom;
        if ( toroidal_domain_location == my_tdl ) {
          partd_comm_.push_back( components[i] );  
        } 
        if ( particle_domain_location == my_pdl ) {
          // record the left gid
          if ( particle_domain_location == hpx_left_pe ) left_pe_ = toroidal_comm_.size(); 
          
          // record the right gid
          if ( particle_domain_location == hpx_right_pe ) right_pe_ = toroidal_comm_.size(); 

          toroidal_comm_.push_back( components[i] );  
        }
      }

      if ( partd_comm_.size() != (std::size_t) npartdom ) {
        std::cerr << " PROBLEM: partd_comm " << partd_comm_.size() 
                     << " != npartdom " << npartdom << std::endl;
      }
      if ( toroidal_comm_.size() != (std::size_t) ntoroidal ) {
        std::cerr << " PROBLEM: toroidal_comm " << toroidal_comm_.size() 
                     << " != ntoroidal " << ntoroidal << std::endl;
      }
    }

    void point::chargei()
    {
      FNAME(chargei)(static_cast<void*>(this));
    }

    void point::partd_allreduce()
    {

      std::cout << " HELLO WORLD FROM allreduce" << std::endl;
//#if 0      
      // synchronize with all operations to finish
      hpx::future<void> f = gate_.get_future();

      std::size_t generation = 0;
      generation = ++generation_;

      double value = item_*3.4159;

      setdata_action setdata;
      hpx::apply(setdata, components_, item_, generation, value);

      // possibly do other stuff while the allgather is going on...
      f.get();
      //gate_.reset();              // reset and-gate
      std::cout << " Finish TEST allreduce " << item_ << std::endl;
//#endif
      //std::size_t length = *mgrid * (*mzetap1);
      //send.resize(length);
      //for (std::size_t ij=0;ij<length;ij++) {
      //  send[ij] = dnitmp[ij];
      //}

      //std::vector<hpx::lcos::future<void> > test_phase;
      //partd_allreduce_receive_action partd_allreduce_receive; 
      //test_phase.push_back(hpx::async(partd_allreduce_receive,components_[0]));
      //hpx::lcos::wait(test_phase);
      //std::cout << " end HELLO WORLD FROM allreduce" << std::endl;

      //partd_allreduce_receive_action partd_allreduce_receive; 
      //for (std::size_t i=0;i<partd_comm_.size();i++) {
       // hpx::apply(partd_allreduce_receive_action(),partd_comm_[i],send,i);
      //  test_phase.push_back(hpx::async(partd_allreduce_receive,partd_comm_[i],send,i));
     // }
     // std::cout << " HELLO WORLD " << std::endl;
     // hpx::lcos::wait(test_phase);
      // synchronize with all operations to finish
      //hpx::future<void> f = gate_.get_future();

      // put other work here
      //f.get();
      //gate_.reset();  // reset and-gate
    }

    void point::allreduce()
    {
      std::cout << " TEST allreduce " << item_ << std::endl;
      // create a new and-gate object
      gate_.init(components_.size());
      
      // synchronize with all operations to finish
      hpx::future<void> f = gate_.get_future();

      std::size_t generation = 0;
      generation = ++generation_;

      double value = item_*3.4159;

      setdata_action setdata;
      hpx::apply(setdata, components_, item_, generation, value);

      // possibly do other stuff while the allgather is going on...
      f.get();
      //gate_.reset();              // reset and-gate
      std::cout << " Finish TEST allreduce " << item_ << std::endl;
     
    }

    void point::setdata(std::size_t item, std::size_t generation, double data)
    {
      while ( generation > generation_ ) {
        hpx::this_thread::suspend();
      }

      std::cout << " TEST setdata " << item_ << " receiving from " << item << std::endl;
      gate_.set(item);
    }

//    void point::partd_allreduce_receive(std::vector<double> const&receive,std::size_t i) 
    void point::partd_allreduce_receive() 
    {
      std::cout << " HELLO WORLD IN receive " << std::endl;
   //   std::cout << " RECEIVED FROM " << i << std::endl;
   //   gate_.set(i);
      return;
    }
}}

