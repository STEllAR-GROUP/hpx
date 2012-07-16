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
            void FNAME(partd_allreduce_cmm) (void* pfoo,double *dnitmp,
                               double *densityi,int *mgrid,int *mzetap1) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = static_cast<gtc::server::point*>(pfoo); 
                    ptr_to_class->partd_allreduce(dnitmp,densityi,mgrid,mzetap1);
                    return; };
}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::setup(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& point_components)
    {
      item_ = mype;
      int t1 = numberpe;
      int t2 = mype;
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      FNAME(setup)(&t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);

      FNAME(load)();

      // Figure out the communicators: toroidal_comm and partd_comm
      int my_pdl = mype%npartdom;
      int my_tdl = mype/npartdom;
      for (std::size_t i=0;i<numberpe;i++) {
        int particle_domain_location= i%npartdom;
        int toroidal_domain_location= i/npartdom;
        if ( toroidal_domain_location == my_tdl ) {
          partd_comm_.push_back( point_components[i] );  
        } 
        if ( particle_domain_location == my_pdl ) {
          // record the left gid
          if ( particle_domain_location == hpx_left_pe ) left_pe_ = toroidal_comm_.size(); 
          
          // record the right gid
          if ( particle_domain_location == hpx_right_pe ) right_pe_ = toroidal_comm_.size(); 

          toroidal_comm_.push_back( point_components[i] );  
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

    void point::partd_allreduce(double *dnitmp,double *densityi,int *mgrid,int *mzetap1)
    {
      std::cout << " HELLO WORLD FROM allreduce" << std::endl;

      // create a new and-gate object
      gate_ = hpx::lcos::local::and_gate(partd_comm_.size());

      std::vector<double> send;
      std::size_t length = *mgrid * (*mzetap1);
      send.resize(length);
      for (std::size_t ij=0;ij<length;ij++) {
        send[ij] = dnitmp[ij];
      }

      for (std::size_t i=0;i<partd_comm_.size();i++) {
        hpx::apply(partd_allreduce_receive_action(),partd_comm_[i],send,i);
      }
      // synchronize with all operations to finish
      hpx::future<void> f = gate_.get_future();

      // put other work here
      f.get();
      gate_.reset();  // reset and-gate
    }

    void point::partd_allreduce_receive(std::vector<double> const&receive,std::size_t i) 
    {
      std::cout << " RECEIVED FROM " << i << std::endl;
      gate_.set(i);
      return;
    }
}}

