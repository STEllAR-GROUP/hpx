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

extern "C" {
            void FNAME(setup_0)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_1)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_2)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_3)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_4)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_5)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_6)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_7)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_8)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(setup_9)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *);
            void FNAME(load)();
            void FNAME(chargei_0)(void* opaque_ptr_to_class);
            void FNAME(chargei_1)(void* opaque_ptr_to_class);
            void FNAME(chargei_2)(void* opaque_ptr_to_class);
            void FNAME(chargei_3)(void* opaque_ptr_to_class);
            void FNAME(chargei_4)(void* opaque_ptr_to_class);
            void FNAME(chargei_5)(void* opaque_ptr_to_class);
            void FNAME(chargei_6)(void* opaque_ptr_to_class);
            void FNAME(chargei_7)(void* opaque_ptr_to_class);
            void FNAME(chargei_8)(void* opaque_ptr_to_class);
            void FNAME(chargei_9)(void* opaque_ptr_to_class);
            void FNAME(sndleft_toroidal_cmm) (void* pfoo,double *send, int* mgrid) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_sndleft(send,mgrid);
                    return; };
            void FNAME(rcvright_toroidal_cmm) (void* pfoo,double *receive) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_rcvright(receive);
                    return; };
            void FNAME(partd_allreduce_cmm) (void* pfoo,double *dnitmp,double *densityi,
                                             int* mgrid, int *mzetap1) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->partd_allreduce(dnitmp,densityi,mgrid,mzetap1);
                    return; };
            void FNAME(broadcast_parameters_cmm) (void* pfoo,
                     int *integer_params,double *real_params,
                     int *n_integers,int *n_reals) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->broadcast_parameters(integer_params,
                              real_params, n_integers,n_reals);
 return; };
}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    void point::setup_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;
      generation_ = 0;
      in_toroidal_ = 0;
      in_particle_ = 0;

      // prepare data array
      //n_.clear();
      //n_.resize(components.size());
#if 0
      // TEST
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      npartdom = 1;
      ntoroidal = 10;
      int particle_domain_location=mype%npartdom;
      int toroidal_domain_location=mype/npartdom;
      int myrank_toroidal = toroidal_domain_location;
      if ( myrank_toroidal > particle_domain_location ) {
        myrank_toroidal = particle_domain_location;
      }
      hpx_left_pe = (myrank_toroidal-1+ntoroidal)%ntoroidal;
      hpx_right_pe = (myrank_toroidal+1)%ntoroidal;
#endif
//#if 0
      int t1 = numberpe;
      int t2 = mype;
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      switch(item_) {
        case 0:
          FNAME(setup_0)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 1:
          FNAME(setup_1)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 2:
          FNAME(setup_2)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 3:
          FNAME(setup_3)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 4:
          FNAME(setup_4)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 5:
          FNAME(setup_5)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 6:
          FNAME(setup_6)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 7:
          FNAME(setup_7)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 8:
          FNAME(setup_8)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
        case 9:
          FNAME(setup_9)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe);
          break;
      }

      //FNAME(load)();
//#endif

      // Figure out the communicators: toroidal_comm and partd_comm
      int my_pdl = mype%npartdom;
      int my_tdl = mype/npartdom;

      if ( my_pdl == (int) mype ) in_particle_ = 1;
      if ( my_tdl == (int) mype ) in_toroidal_ = 1;

      left_pe_ = mype - 1;
      right_pe_ = mype + 1;
      if ( left_pe_ < 0 ) left_pe_ = components_.size()-1;
      if ( right_pe_ > components_.size()-1 ) right_pe_ = 0;
      toroidal_comm_ = components_;

      for (std::size_t i=0;i<numberpe;i++) {
        int particle_domain_location= i%npartdom;
        int toroidal_domain_location= i/npartdom;
        if ( toroidal_domain_location == my_tdl ) {
          partd_comm_.push_back( components[i] );
        }
        //if ( particle_domain_location == my_pdl ) {
          // record the left gid
          //if ( particle_domain_location == hpx_left_pe ) left_pe_ = toroidal_comm_.size();

          // record the right gid
          //if ( particle_domain_location == hpx_right_pe ) right_pe_ = toroidal_comm_.size();
        //  toroidal_comm_.push_back( components[i] );
       // }
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

    void point::chargei_wrapper()
    {
      switch(item_) {
        case 0:
          FNAME(chargei_0)(static_cast<void*>(this));
          break;
        case 1:
          FNAME(chargei_1)(static_cast<void*>(this));
          break;
        case 2:
          FNAME(chargei_2)(static_cast<void*>(this));
          break;
        case 3:
          FNAME(chargei_3)(static_cast<void*>(this));
          break;
        case 4:
          FNAME(chargei_4)(static_cast<void*>(this));
          break;
        case 5:
          FNAME(chargei_5)(static_cast<void*>(this));
          break;
        case 6:
          FNAME(chargei_6)(static_cast<void*>(this));
          break;
        case 7:
          FNAME(chargei_7)(static_cast<void*>(this));
          break;
        case 8:
          FNAME(chargei_8)(static_cast<void*>(this));
          break;
        case 9:
          FNAME(chargei_9)(static_cast<void*>(this));
          break;
      }
    }

    void point::broadcast_parameters(int *integer_params,double *real_params,
                             int *n_integers,int *n_reals)
    {
      int nint = *n_integers;
      int nreal = *n_reals;

      if ( item_ != 0 ) {
        // create a new and-gate object
        gate_.init(1);

        // synchronize with all operations to finish
        hpx::future<void> f = gate_.get_future();

        {
          mutex_type::scoped_lock l(mtx_);
          ++generation_;
        }

        f.get();

        // Copy the parameters to the fortran arrays
        for (int i=0;i<intparams_.size();i++) {
          integer_params[i] = intparams_[i];
        }
        for (int i=0;i<realparams_.size();i++) {
          real_params[i] = realparams_[i];
        }
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = 0;
        {
          mutex_type::scoped_lock l(mtx_);
          generation = ++generation_;
        }
        std::vector<int> intparams;
        std::vector<double> realparams;
        intparams.resize(nint);
        for (int i=0;i<nint;i++) {
          intparams[i] = integer_params[i];
        }
        realparams.resize(nreal);
        for (int i=0;i<nreal;i++) {
          realparams[i] = real_params[i];
        }

        // eliminate item 0's (the sender's) gid
        std::vector<hpx::naming::id_type> all_but_root;
        all_but_root.resize(components_.size()-1);
        for (std::size_t i=0;i<all_but_root.size();i++) {
          all_but_root[i] = components_[i+1];
        }

        set_params_action set_params_;
        hpx::apply(set_params_, all_but_root, item_, generation,
                        intparams,realparams);
      }
    }

    void point::set_params(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& intparams,
                           std::vector<double> const& realparams)
    {
       mutex_type::scoped_lock l(mtx_);

       // make sure this set operation has not arrived ahead of time
       while (generation > generation_)
       {
         hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
         hpx::this_thread::suspend();
       }

       intparams_ = intparams;
       realparams_ = realparams;

       gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::partd_allreduce(double *dnitmp,double *densityi, int* mgrid, int *mzetap1)
    {
      if ( in_particle_ ) {
        // create a new and-gate object
        gate_.init(partd_comm_.size());

        // synchronize with all operations to finish
        hpx::future<void> f = gate_.get_future();

        std::size_t generation = 0;
        {
          mutex_type::scoped_lock l(mtx_);
          generation = ++generation_;
        }

        int vsize = (*mgrid)*(*mzetap1);
        std::vector<double> dnisend;
        dnisend.resize(vsize); 
        dnireceive_.resize(vsize); 
        std::fill( dnireceive_.begin(),dnireceive_.end(),0.0);

        for (std::size_t i=0;i<dnisend.size();i++) {
          dnisend[i] = dnitmp[i];
        }

        set_data_action set_data_;
        hpx::apply(set_data_, partd_comm_, item_, generation, dnisend);

        // possibly do other stuff while the allgather is going on...
        f.get();

        for (std::size_t i=0;i<dnireceive_.size();i++) {
          densityi[i] = dnireceive_[i]; 
        }
      }
    }

    void point::set_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
       mutex_type::scoped_lock l(mtx_);

       // make sure this set operation has not arrived ahead of time
       while (generation > generation_)
       {
         hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
         hpx::this_thread::suspend();
       }

       for (std::size_t i=0;i<dnireceive_.size();i++) {
         dnireceive_[i] += data[i];
       }

       gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::toroidal_sndleft(double *csend,int* mgrid)
    {
      if ( in_toroidal_ ) {
        // Send data to the left
        // The sender: send data to the left 
        // in a fire and forget fashion
        int vsize = *mgrid;
        std::vector<double> send;
        send.resize(vsize); 
        tsr_receive_.resize(vsize); 

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        std::size_t generation = 0;
        {
          mutex_type::scoped_lock l(mtx_);
          generation = ++generation_;
        }

        set_tsr_data_action set_tsr_data_;
        hpx::apply(set_tsr_data_, toroidal_comm_[left_pe_], item_, generation, send);
      }
    }

    void point::toroidal_rcvright(double *creceive)
    {
      std::cout << " TEST A " << item_ << std::endl;
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        // create a new and-gate object
        gate_.init(1);

        // synchronize with all operations to finish
        hpx::future<void> f = gate_.get_future();

        {
          mutex_type::scoped_lock l(mtx_);
          ++generation_;
        }

        std::cout << " TEST item " << item_ << std::endl;

        // possibly do other stuff 
        f.get();

        std::cout << " TEST item " << item_ << std::endl;
        for (std::size_t i=0;i<tsr_receive_.size();i++) {
          creceive[i] = tsr_receive_[i]; 
        }
      }
    }

    void point::set_tsr_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
       mutex_type::scoped_lock l(mtx_);
       std::cout << " TEST which " << which << " item " << item_ << std::endl;

       // make sure this set operation has not arrived ahead of time
       while (generation > generation_)
       {
         hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
         hpx::this_thread::suspend();
       }

       tsr_receive_ = send;

       gate_.set(0);         // trigger corresponding and-gate input
    }

}}

