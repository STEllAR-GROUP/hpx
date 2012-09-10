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
            void FNAME(load_0)();
            void FNAME(load_1)();
            void FNAME(load_2)();
            void FNAME(load_3)();
            void FNAME(load_4)();
            void FNAME(load_5)();
            void FNAME(load_6)();
            void FNAME(load_7)();
            void FNAME(load_8)();
            void FNAME(load_9)();
            void FNAME(setup_0)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_1)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_2)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_3)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_4)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_5)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_6)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_7)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_8)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
            void FNAME(setup_9)(void* opaque_ptr_to_class,
                          int *,int *,int *,int *,int *, int *,int *);
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
            void FNAME(smooth_0)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_1)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_2)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_3)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_4)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_5)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_6)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_7)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_8)(void* opaque_ptr_to_class,int *);
            void FNAME(smooth_9)(void* opaque_ptr_to_class,int *);
            void FNAME(fieldr_0)(void* opaque_ptr_to_class);
            void FNAME(fieldr_1)(void* opaque_ptr_to_class);
            void FNAME(fieldr_2)(void* opaque_ptr_to_class);
            void FNAME(fieldr_3)(void* opaque_ptr_to_class);
            void FNAME(fieldr_4)(void* opaque_ptr_to_class);
            void FNAME(fieldr_5)(void* opaque_ptr_to_class);
            void FNAME(fieldr_6)(void* opaque_ptr_to_class);
            void FNAME(fieldr_7)(void* opaque_ptr_to_class);
            void FNAME(fieldr_8)(void* opaque_ptr_to_class);
            void FNAME(fieldr_9)(void* opaque_ptr_to_class);
            void FNAME(pushi_0)(void* opaque_ptr_to_class);
            void FNAME(pushi_1)(void* opaque_ptr_to_class);
            void FNAME(pushi_2)(void* opaque_ptr_to_class);
            void FNAME(pushi_3)(void* opaque_ptr_to_class);
            void FNAME(pushi_4)(void* opaque_ptr_to_class);
            void FNAME(pushi_5)(void* opaque_ptr_to_class);
            void FNAME(pushi_6)(void* opaque_ptr_to_class);
            void FNAME(pushi_7)(void* opaque_ptr_to_class);
            void FNAME(pushi_8)(void* opaque_ptr_to_class);
            void FNAME(pushi_9)(void* opaque_ptr_to_class);
            void FNAME(shifti_0)(void* opaque_ptr_to_class);
            void FNAME(shifti_1)(void* opaque_ptr_to_class);
            void FNAME(shifti_2)(void* opaque_ptr_to_class);
            void FNAME(shifti_3)(void* opaque_ptr_to_class);
            void FNAME(shifti_4)(void* opaque_ptr_to_class);
            void FNAME(shifti_5)(void* opaque_ptr_to_class);
            void FNAME(shifti_6)(void* opaque_ptr_to_class);
            void FNAME(shifti_7)(void* opaque_ptr_to_class);
            void FNAME(shifti_8)(void* opaque_ptr_to_class);
            void FNAME(shifti_9)(void* opaque_ptr_to_class);
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
            void FNAME(sndright_toroidal_cmm) (void* pfoo,double *send, int* mgrid) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_sndright(send,mgrid);
                    return; };
            void FNAME(rcvleft_toroidal_cmm) (void* pfoo,double *receive) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_rcvleft(receive);
                    return; };
            void FNAME(partd_allreduce_cmm) (void* pfoo,double *dnitmp,double *densityi,
                                             int* mgrid, int *mzetap1) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->partd_allreduce(dnitmp,densityi,mgrid,mzetap1);
                    return; };
            void FNAME(toroidal_allreduce_cmm) (void* pfoo,double *input,double *output,
                                             int* size) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_allreduce(input,output,size);
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
            void FNAME(toroidal_gather_cmm) (void* pfoo,double *input,
                                             int* size,int* dst) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_gather(input,size,dst);
                    return; };
            void FNAME(toroidal_gather_receive_cmm) (void* pfoo,double *output,
                                                     int* dst) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_gather_receive(output,dst);
                    return; };
            void FNAME(toroidal_scatter_cmm) (void* pfoo,double *input,
                                             int* size,int* src) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_scatter(input,size,src);
                    return; };
            void FNAME(toroidal_scatter_receive_cmm) (void* pfoo,double *output,
                                                     int* dst) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->toroidal_scatter_receive(output,dst);
                    return; };
            void FNAME(comm_allreduce_cmm) (void* pfoo,double *in,double *out,
                                             int* msize) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->comm_allreduce(in,out,msize);
                    return; };
            void FNAME(int_comm_allreduce_cmm) (void* pfoo,int *in,int *out,
                                             int* msize) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->int_comm_allreduce(in,out,msize);
                    return; };
            void FNAME(int_sndright_toroidal_cmm) (void* pfoo,int *send, int* mgrid) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->int_toroidal_sndright(send,mgrid);
                    return; };
            void FNAME(int_rcvleft_toroidal_cmm) (void* pfoo,int *receive) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->int_toroidal_rcvleft(receive);
                    return; };
            void FNAME(int_sndleft_toroidal_cmm) (void* pfoo,int *send, int* mgrid) {
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->int_toroidal_sndleft(send,mgrid);
                    return; };
            void FNAME(int_rcvright_toroidal_cmm) (void* pfoo,int *receive) {
                    // Cast to gtc::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtc::server::point *ptr_to_class = *static_cast<gtc::server::point**>(pfoo);
                    ptr_to_class->int_toroidal_rcvright(receive);
                    return; };
}

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
namespace gtc { namespace server
{
    std::size_t point::setup_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;
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
      hpx_left_pe = (myrank_toroidal-1+ntoroidal)%ntoroidal;
      hpx_right_pe = (myrank_toroidal+1)%ntoroidal;
#endif
//#if 0
      int t1 = static_cast<int>(numberpe);
      int t2 = static_cast<int>(mype);
      int npartdom,ntoroidal;
      int hpx_left_pe, hpx_right_pe;
      int mstep;
      switch(item_) {
        case 0:
          FNAME(setup_0)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 1:
          FNAME(setup_1)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 2:
          FNAME(setup_2)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 3:
          FNAME(setup_3)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 4:
          FNAME(setup_4)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 5:
          FNAME(setup_5)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 6:
          FNAME(setup_6)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 7:
          FNAME(setup_7)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 8:
          FNAME(setup_8)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
        case 9:
          FNAME(setup_9)(static_cast<void*>(this),
            &t1,&t2,&npartdom,&ntoroidal,&hpx_left_pe,&hpx_right_pe,&mstep);
          break;
      }

      switch(item_) {
        case 0:
          FNAME(load_0)();
          break;
        case 1:
          FNAME(load_1)();
          break;
        case 2:
          FNAME(load_2)();
          break;
        case 3:
          FNAME(load_3)();
          break;
        case 4:
          FNAME(load_4)();
          break;
        case 5:
          FNAME(load_5)();
          break;
        case 6:
          FNAME(load_6)();
          break;
        case 7:
          FNAME(load_7)();
          break;
        case 8:
          FNAME(load_8)();
          break;
        case 9:
          FNAME(load_9)();
          break;
      }
//#endif

      // Figure out the communicators: toroidal_comm and partd_comm
      std::size_t my_pdl = mype%npartdom;
      std::size_t my_tdl = mype/npartdom;

      if ( my_pdl == mype ) in_particle_ = 1;
      if ( my_tdl == mype ) in_toroidal_ = 1;

      for (std::size_t i=0;i<numberpe;i++) {
        std::size_t particle_domain_location = i%npartdom;
        std::size_t toroidal_domain_location = i/npartdom;

        if ( particle_domain_location == my_pdl ) {
          toroidal_comm_.push_back(components_[i]);
        }

        if ( toroidal_domain_location == my_tdl ) {
          partd_comm_.push_back(components_[i]);
        }
      }

      left_pe_ = hpx_left_pe;
      right_pe_ = hpx_right_pe;

      if ( partd_comm_.size() != (std::size_t) npartdom ) {
        std::cerr << " PROBLEM: partd_comm " << partd_comm_.size()
                     << " != npartdom " << npartdom << std::endl;
      }
      if ( toroidal_comm_.size() != (std::size_t) ntoroidal ) {
        std::cerr << " PROBLEM: toroidal_comm " << toroidal_comm_.size()
                     << " != ntoroidal " << ntoroidal << std::endl;
      }

      std::size_t tmp = (std::size_t) mstep;
      return tmp;
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
        // synchronize with all operations to finish
        hpx::future<void> f = broadcast_gate_.get_future(1);

        f.get();

        // Copy the parameters to the fortran arrays
        for (std::size_t i=0;i<intparams_.size();i++) {
          integer_params[i] = intparams_[i];
        }
        for (std::size_t i=0;i<realparams_.size();i++) {
          real_params[i] = realparams_[i];
        }
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = broadcast_gate_.next_generation();

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
        for (std::size_t i=0;i<all_but_root.size();i++) {
          hpx::apply(set_params_, all_but_root[i], item_, generation,
                     intparams,realparams);
        }
      }
    }

    void point::set_params(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& intparams,
                           std::vector<double> const& realparams)
    {
        broadcast_gate_.synchronize(generation, "point::set_params");

        {
            mutex_type::scoped_lock l(mtx_);
            intparams_ = intparams;
            realparams_ = realparams;
        }

        broadcast_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::partd_allreduce(double *dnitmp,double *densityi, int* mgrid, int *mzetap1)
    {
      if ( in_particle_ ) {
        int vsize = (*mgrid)*(*mzetap1);
        dnireceive_.resize(vsize);

        // synchronize with all operations to finish
        hpx::future<void> f = allreduce_gate_.get_future(partd_comm_.size());
        std::size_t generation = allreduce_gate_.generation();

        std::vector<double> dnisend;
        dnisend.resize(vsize);
        std::fill( dnireceive_.begin(),dnireceive_.end(),0.0);

        for (std::size_t i=0;i<dnisend.size();i++) {
          dnisend[i] = dnitmp[i];
        }

        set_data_action set_data_;
        for (std::size_t i=0;i<partd_comm_.size();i++) {
          hpx::apply(set_data_, partd_comm_[i], item_, generation, dnisend);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<dnireceive_.size();i++) {
          densityi[i] = dnireceive_[i];
        }
      } else {
        allreduce_gate_.next_generation();
      }
    }

    void point::set_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_data");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<dnireceive_.size();i++)
                dnireceive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::toroidal_sndleft(double *csend,int* mgrid)
    {
//       std::cout << "toroidal_sndleft: " << item_ << " -> " << left_pe_
//                 << " (g: " << sndleft_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        int vsize = *mgrid;
        sendleft_receive_.resize(vsize);

        // create a new and-gate object
        sndleft_future_ = sndleft_gate_.get_future(1);
        std::size_t generation = sndleft_gate_.generation();

        // Send data to the left
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<double> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        set_sendleft_data_action set_sendleft_data_;
        hpx::apply(set_sendleft_data_, toroidal_comm_[left_pe_], item_,
            generation, send);
      } else {
        sndleft_gate_.next_generation();
      }
    }

    void point::toroidal_rcvright(double *creceive)
    {
//       std::cout << "toroidal_rcvleft: " << item_
//                 << " (g: " << generation_ << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        // Now receive a message from the right
        sndleft_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<sendleft_receive_.size();i++) {
          creceive[i] = sendleft_receive_[i];
        }
      }
    }

    void point::set_sendleft_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
//         std::cout << "set_sendleft_data: " << item_ << " <- " << which
//                 << " (g: " << sndleft_gate_.generation() << ", " << generation << ")"
//                 << std::endl;

        sndleft_gate_.synchronize(generation, "point::set_sendleft_data");

        {
            mutex_type::scoped_lock l(mtx_);
            sendleft_receive_ = send;
        }

        sndleft_gate_.set(0);         // trigger corresponding and-gate input
    }

    void point::toroidal_allreduce(double *input,double *output, int* size)
    {
      if ( in_toroidal_ ) {
        int vsize = *size;
        treceive_.resize(vsize);

        // synchronize with all operations to finish
        hpx::future<void> f = allreduce_gate_.get_future(toroidal_comm_.size());
        std::size_t generation = allreduce_gate_.generation();

        std::vector<double> send;
        send.resize(vsize);
        std::fill( treceive_.begin(),treceive_.end(),0.0);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = input[i];
        }

        set_tdata_action set_tdata_;
        for (std::size_t i=0;i<toroidal_comm_.size();i++) {
          hpx::apply(set_tdata_, toroidal_comm_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<treceive_.size();i++) {
          output[i] = treceive_[i];
        }
      } else {
        allreduce_gate_.next_generation();
      }
    }

    void point::set_tdata(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_tdata");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<treceive_.size();i++)
                treceive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::timeloop(std::size_t istep, std::size_t irk)
    {
      set_description("smooth");
//       std::cout << "smooth: " << item_ << std::endl;

      // Call smooth(3) {{{
      int flag = 3;
      switch(item_) {
        case 0:
          FNAME(smooth_0)(static_cast<void*>(this),&flag);
          break;
        case 1:
          FNAME(smooth_1)(static_cast<void*>(this),&flag);
          break;
        case 2:
          FNAME(smooth_2)(static_cast<void*>(this),&flag);
          break;
        case 3:
          FNAME(smooth_3)(static_cast<void*>(this),&flag);
          break;
        case 4:
          FNAME(smooth_4)(static_cast<void*>(this),&flag);
          break;
        case 5:
          FNAME(smooth_5)(static_cast<void*>(this),&flag);
          break;
        case 6:
          FNAME(smooth_6)(static_cast<void*>(this),&flag);
          break;
        case 7:
          FNAME(smooth_7)(static_cast<void*>(this),&flag);
          break;
        case 8:
          FNAME(smooth_8)(static_cast<void*>(this),&flag);
          break;
        case 9:
          FNAME(smooth_9)(static_cast<void*>(this),&flag);
          break;
      }
      // }}}

      set_description("field");
//       std::cout << "field: " << item_ << std::endl;

      // Call field {{{
      switch(item_) {
        case 0:
          FNAME(fieldr_0)(static_cast<void*>(this));
          break;
        case 1:
          FNAME(fieldr_1)(static_cast<void*>(this));
          break;
        case 2:
          FNAME(fieldr_2)(static_cast<void*>(this));
          break;
        case 3:
          FNAME(fieldr_3)(static_cast<void*>(this));
          break;
        case 4:
          FNAME(fieldr_4)(static_cast<void*>(this));
          break;
        case 5:
          FNAME(fieldr_5)(static_cast<void*>(this));
          break;
        case 6:
          FNAME(fieldr_6)(static_cast<void*>(this));
          break;
        case 7:
          FNAME(fieldr_7)(static_cast<void*>(this));
          break;
        case 8:
          FNAME(fieldr_8)(static_cast<void*>(this));
          break;
        case 9:
          FNAME(fieldr_9)(static_cast<void*>(this));
          break;
      }
      // }}}

      set_description("pushi");
//       std::cout << "pushi: " << item_ << std::endl;

      // Call pushi {{{
      switch(item_) {
        case 0:
          FNAME(pushi_0)(static_cast<void*>(this));
          break;
        case 1:
          FNAME(pushi_1)(static_cast<void*>(this));
          break;
        case 2:
          FNAME(pushi_2)(static_cast<void*>(this));
          break;
        case 3:
          FNAME(pushi_3)(static_cast<void*>(this));
          break;
        case 4:
          FNAME(pushi_4)(static_cast<void*>(this));
          break;
        case 5:
          FNAME(pushi_5)(static_cast<void*>(this));
          break;
        case 6:
          FNAME(pushi_6)(static_cast<void*>(this));
          break;
        case 7:
          FNAME(pushi_7)(static_cast<void*>(this));
          break;
        case 8:
          FNAME(pushi_8)(static_cast<void*>(this));
          break;
        case 9:
          FNAME(pushi_9)(static_cast<void*>(this));
          break;
      }
      // }}}

      set_description("shifti");
//       std::cout << "shifti: " << item_ << std::endl;

#if 0
      // Call shifti {{{
      switch(item_) {
        case 0:
          FNAME(shifti_0)(static_cast<void*>(this));
          break;
        case 1:
          FNAME(shifti_1)(static_cast<void*>(this));
          break;
        case 2:
          FNAME(shifti_2)(static_cast<void*>(this));
          break;
        case 3:
          FNAME(shifti_3)(static_cast<void*>(this));
          break;
        case 4:
          FNAME(shifti_4)(static_cast<void*>(this));
          break;
        case 5:
          FNAME(shifti_5)(static_cast<void*>(this));
          break;
        case 6:
          FNAME(shifti_6)(static_cast<void*>(this));
          break;
        case 7:
          FNAME(shifti_7)(static_cast<void*>(this));
          break;
        case 8:
          FNAME(shifti_8)(static_cast<void*>(this));
          break;
        case 9:
          FNAME(shifti_9)(static_cast<void*>(this));
          break;
      }
      // }}}
#endif
    }

    void point::toroidal_sndright(double *csend,int* mgrid)
    {
//       std::cout << "toroidal_sndright: " << item_ << " -> " << right_pe_
//                 << " (g: " << sndright_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        // create a new and-gate object
        int vsize = *mgrid;
        sendright_receive_.resize(vsize);

        sndright_future_ = sndright_gate_.get_future(1);
        std::size_t generation = sndright_gate_.generation();

        // Send data to the right
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<double> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        set_sendright_data_action set_sendright_data_;
        hpx::apply(set_sendright_data_,
             toroidal_comm_[right_pe_], item_, generation, send);
      } else {
        sndright_gate_.next_generation();
      }
    }

    void point::toroidal_rcvleft(double *creceive)
    {
//       std::cout << "toroidal_rcvleft: " << item_
//                 << " (g: " << generation_ << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        // Now receive a message from the right
        sndright_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<sendright_receive_.size();i++) {
          creceive[i] = sendright_receive_[i];
        }
      }
    }

    void point::set_sendright_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
//         std::cout << "set_sendright_data: " << item_ << " <- " << which
//                 << " (g: " << sndright_gate_.generation() << ", " << generation << ")"
//                 << std::endl;

        sndright_gate_.synchronize(generation, "point::set_sendright_data");

        {
            mutex_type::scoped_lock l(mtx_);
            sendright_receive_ = send;
        }

        sndright_gate_.set(0);         // trigger corresponding and-gate input
    }

    void point::toroidal_gather(double *csend, int *tsize,int *tdst)
    {
//       std::cout << "toroidal_gather: " << item_
//                 << " (g: " << gather_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        int vsize = *tsize;

        // create a new and-gate object
        std::size_t generation = 0;
        if ( *tdst == item_ ) {
          toroidal_gather_receive_.resize(toroidal_comm_.size()*vsize);
          gather_future_ = gather_gate_.get_future(toroidal_comm_.size());
          generation = gather_gate_.generation();
        }
        else {
          generation = gather_gate_.next_generation();
        }

        // Send data to dst
        // The sender: send data to the left
        // in a fire and forget fashion
        int dst = *tdst;
        std::vector<double> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        set_toroidal_gather_data_action set_toroidal_gather_data_;
        hpx::apply(set_toroidal_gather_data_,
             toroidal_comm_[dst], item_, generation, send);
      } else {
        gather_gate_.next_generation();
      }

    }

    void point::toroidal_gather_receive(double *creceive, int *cdst)
    {
      int dst = *cdst;
      if ( dst == item_ ) {
        // synchronize with all operations to finish
        gather_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<toroidal_gather_receive_.size();i++) {
          creceive[i] = toroidal_gather_receive_[i];
        }
      }
    }

    void point::set_toroidal_gather_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
//         std::cout << "set_toroidal_gather_data: " << item_ << " <- " << which
//                 << " (g: " << gather_gate_.generation() << ", " << generation << ")"
//                 << std::endl;

        gather_gate_.synchronize(generation, "point::set_toroidal_gather_data");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<send.size();i++)
                toroidal_gather_receive_[which*send.size()+i] = send[i];
        }

        gather_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::toroidal_scatter(double *csend, int *tsize,int *tsrc)
    {
      int src = *tsrc;
      if ( src == item_ ) {
        // create a new and-gate object
        int vsize = *tsize;
        toroidal_scatter_receive_.resize(vsize);

        scatter_future_ = scatter_gate_.get_future(1);
        std::size_t generation = scatter_gate_.generation();

        // Send data to everyone in toroidal
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<double> send;
        send.resize(vsize);

        for (std::size_t step=0;step<toroidal_comm_.size();step++) {
          for (std::size_t i=0;i<send.size();i++) {
            send[i] = csend[i+step*vsize];
          }

          set_toroidal_scatter_data_action set_toroidal_scatter_data_;
          for (std::size_t i=0;i<toroidal_comm_.size();i++) {
            hpx::apply(set_toroidal_scatter_data_,
               toroidal_comm_[i], item_, generation, send);
          }
        }
      } else {
        scatter_gate_.next_generation();
      }

    }

    void point::toroidal_scatter_receive(double *creceive, int *cdst)
    {
      if ( in_toroidal_ ) {
        // synchronize with all operations to finish
        scatter_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<toroidal_scatter_receive_.size();i++) {
          creceive[i] = toroidal_scatter_receive_[i];
        }
      }
    }

    void point::set_toroidal_scatter_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        scatter_gate_.synchronize(generation, "point::set_toroidal_scatter_data");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<send.size();i++)
                toroidal_scatter_receive_[i] = send[i];
        }

        scatter_gate_.set(0);         // trigger corresponding and-gate input
    }

    void point::comm_allreduce(double *in,double *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        comm_allreduce_receive_.resize(vsize);

        hpx::future<void> f = allreduce_gate_.get_future(components_.size());
        std::size_t generation = allreduce_gate_.generation();

        std::vector<double> send;
        send.resize(vsize);
        std::fill( comm_allreduce_receive_.begin(),comm_allreduce_receive_.end(),0.0);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = in[i];
        }

        set_comm_allreduce_data_action set_data_;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_data_, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<comm_allreduce_receive_.size();i++) {
          out[i] = comm_allreduce_receive_[i];
        }
    }

    void point::set_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_comm_allreduce_data");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<comm_allreduce_receive_.size();i++)
                comm_allreduce_receive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::int_comm_allreduce(int *in,int *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        int_comm_allreduce_receive_.resize(vsize);

        hpx::future<void> f = allreduce_gate_.get_future(components_.size());
        std::size_t generation = allreduce_gate_.generation();

        std::vector<int> send;
        send.resize(vsize);
        std::fill( int_comm_allreduce_receive_.begin(),int_comm_allreduce_receive_.end(),0);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = in[i];
        }

        set_int_comm_allreduce_data_action set_data_;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_data_, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<comm_allreduce_receive_.size();i++) {
          out[i] = int_comm_allreduce_receive_[i];
        }
    }

    void point::set_int_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_int_comm_allreduce_data");

        {
            mutex_type::scoped_lock l(mtx_);
            for (std::size_t i=0;i<int_comm_allreduce_receive_.size();i++)
                int_comm_allreduce_receive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void point::int_toroidal_sndright(int *csend,int* mgrid)
    {
      if ( in_toroidal_ ) {
        int vsize = *mgrid;
        int_sendright_receive_.resize(vsize);

        // create a new and-gate object
        sndright_future_ = sndright_gate_.get_future(1);
        std::size_t generation = sndright_gate_.generation();

        // Send data to the right
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<int> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        set_int_sendright_data_action set_int_sendright_data_;
        hpx::apply(set_int_sendright_data_,
             toroidal_comm_[right_pe_], item_, generation, send);
      } else {
        sndright_gate_.next_generation();
      }
    }

    void point::int_toroidal_rcvleft(int *creceive)
    {
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        sndright_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<int_sendright_receive_.size();i++) {
          creceive[i] = int_sendright_receive_[i];
        }
      }
    }

    void point::set_int_sendright_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send)
    {
        sndright_gate_.synchronize(generation, "point::set_int_sendright_data");

        {
            mutex_type::scoped_lock l(mtx_);
            int_sendright_receive_ = send;
        }

        sndright_gate_.set(0);         // trigger corresponding and-gate input
    }

    void point::int_toroidal_sndleft(int *csend,int* mgrid)
    {
      if ( in_toroidal_ ) {
        int vsize = *mgrid;
        int_sendleft_receive_.resize(vsize);

        // create a new and-gate object
        sndleft_future_ = sndleft_gate_.get_future(1);
        std::size_t generation = sndleft_gate_.generation();

        // Send data to the left
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<int> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = csend[i];
        }

        set_int_sendleft_data_action set_int_sendleft_data_;
        hpx::apply(set_int_sendleft_data_, toroidal_comm_[left_pe_], item_, generation, send);
      } else {
        sndleft_gate_.next_generation();
      }
    }

    void point::int_toroidal_rcvright(int *creceive)
    {
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        sndleft_future_.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<int_sendleft_receive_.size();i++) {
          creceive[i] = int_sendleft_receive_[i];
        }
      }
    }

    void point::set_int_sendleft_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send)
    {
        sndleft_gate_.synchronize(generation, "point::set_int_sendleft_data");

        {
            mutex_type::scoped_lock l(mtx_);
            int_sendleft_receive_ = send;
        }

        sndleft_gate_.set(0);         // trigger corresponding and-gate input
    }

}}

