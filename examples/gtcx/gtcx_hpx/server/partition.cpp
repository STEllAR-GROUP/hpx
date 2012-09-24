//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>

#include <hpx/lcos/future_wait.hpp>

#include "../../fname.h"
#include "partition.hpp"

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

extern "C" {
            void FNAME(int_allgather_cmm) (void *pfoo,int *in,int *out,int *length) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_comm_allgather(in,out,length);
                    return; };
            void FNAME(set_partd_cmm) (void* pfoo,int *send, int* length) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->set_partd_cmm(send,length);
                    return; };
            void FNAME(set_toroidal_cmm) (void* pfoo,int *send, int* length) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->set_toroidal_cmm(send,length);
                    return; };
            void FNAME(loop)(void* opaque_ptr_to_class, int *,int *);
            void FNAME(sndleft_toroidal_cmm) (void* pfoo,double *send, int* mgrid) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_sndleft(send,mgrid);
                    return; };
            void FNAME(rcvright_toroidal_cmm) (void* pfoo,double *receive) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_rcvright(receive);
                    return; };
            void FNAME(sndright_toroidal_cmm) (void* pfoo,double *send, int* mgrid) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_sndright(send,mgrid);
                    return; };
            void FNAME(rcvleft_toroidal_cmm) (void* pfoo,double *receive) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_rcvleft(receive);
                    return; };
            void FNAME(partd_allreduce_cmm) (void* pfoo,double *dnitmp,double *densityi,
                                             int* mgrid, int *mzetap1) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->partd_allreduce(dnitmp,densityi,mgrid,mzetap1);
                    return; };
            void FNAME(toroidal_allreduce_cmm) (void* pfoo,double *input,double *output,
                                             int* size) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_allreduce(input,output,size);
                    return; };
            void FNAME(broadcast_parameters_cmm) (void* pfoo,
                     int *integer_params,double *real_params,
                     int *n_integers,int *n_reals) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->broadcast_parameters(integer_params,
                              real_params, n_integers,n_reals);
                    return; };
            void FNAME(toroidal_gather_cmm) (void* pfoo,double *input,
                                             int* size,int* dst) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_gather(input,size,dst);
                    return; };
            void FNAME(toroidal_gather_receive_cmm) (void* pfoo,double *output,
                                                     int* dst) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_gather_receive(output,dst);
                    return; };
            void FNAME(toroidal_scatter_cmm) (void* pfoo,double *input,
                                             int* size,int* src) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_scatter(input,size,src);
                    return; };
            void FNAME(toroidal_scatter_receive_cmm) (void* pfoo,double *output,
                                                     int* dst) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_scatter_receive(output,dst);
                    return; };
            void FNAME(comm_allreduce_cmm) (void* pfoo,double *in,double *out,
                                             int* msize) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->comm_allreduce(in,out,msize);
                    return; };
            void FNAME(int_comm_allreduce_cmm) (void* pfoo,int *in,int *out,
                                             int* msize) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_comm_allreduce(in,out,msize);
                    return; };
            void FNAME(int_sndright_toroidal_cmm) (void* pfoo,int *send, int* mgrid) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_toroidal_sndright(send,mgrid);
                    return; };
            void FNAME(int_rcvleft_toroidal_cmm) (void* pfoo,int *receive) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_toroidal_rcvleft(receive);
                    return; };
            void FNAME(int_sndleft_toroidal_cmm) (void* pfoo,int *send, int* mgrid) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_toroidal_sndleft(send,mgrid);
                    return; };
            void FNAME(int_rcvright_toroidal_cmm) (void* pfoo,int *receive) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_toroidal_rcvright(receive);
                    return; };
}

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
namespace gtcx { namespace server
{
    void partition::loop_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;
      in_toroidal_ = 0;
      in_particle_ = 0;

      int t2 = static_cast<int>(numberpe);
      int t1 = static_cast<int>(mype);

      FNAME(loop)(static_cast<void*>(this), &t1,&t2);
    }

    void partition::broadcast_parameters(int *integer_params,double *real_params,
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

    void partition::set_params(std::size_t which,
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

    void partition::partd_allreduce(double *dnitmp,double *densityi, int* mgrid, int *mzetap1)
    {
      if ( in_particle_ ) {
        int vsize = (*mgrid)*(*mzetap1);
        dnireceive_.resize(vsize);

        // synchronize with all operations to finish
        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(partd_comm_.size(),
            &generation);

        std::vector<double> dnisend;
        dnisend.resize(vsize, 0.0);

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

    void partition::set_data(std::size_t which,
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

    void partition::toroidal_sndleft(double *csend,int* mgrid)
    {
//       std::cout << "toroidal_sndleft: " << item_ << " -> " << left_pe_
//                 << " (g: " << sndleft_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        // create a new and-gate object
        std::size_t generation = 0;
        std::vector<double> send;

        {
            mutex_type::scoped_lock l(mtx_);

            int vsize = *mgrid;
            sendleft_receive_.resize(vsize);
            sndleft_future_ = sndleft_gate_.get_future(&generation);

            // Send data to the left
            // The sender: send data to the left
            // in a fire and forget fashion
            send.resize(vsize);

            for (std::size_t i=0;i<send.size();i++) {
              send[i] = csend[i];
            }
        }

        set_sendleft_data_action set_sendleft_data_;
        hpx::apply(set_sendleft_data_, toroidal_comm_[left_pe_], item_,
            generation, send);
      } else {
        mutex_type::scoped_lock l(mtx_);
        sndleft_gate_.next_generation();
      }
    }

    void partition::toroidal_rcvright(double *creceive)
    {
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        mutex_type::scoped_lock l(mtx_);
        hpx::future<void> f = sndleft_future_;

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        for (std::size_t i=0;i<sendleft_receive_.size();i++) {
          creceive[i] = sendleft_receive_[i];
        }
//         std::cout << "toroidal_rcvright: " << item_
//                   << " (g: " << sndleft_gate_.generation() << ")"
//                   << std::endl;
      }
    }

    void partition::set_sendleft_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
//         std::cout << "set_sendleft_data: " << item_ << " <- " << which
//                 << " (g: " << sndleft_gate_.generation() << ", " << generation << ")"
//                 << std::endl;

        mutex_type::scoped_lock l(mtx_);
        sndleft_gate_.synchronize(generation, l, "point::set_sendleft_data");
        sendleft_receive_ = send;
        sndleft_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::toroidal_allreduce(double *input,double *output, int* size)
    {
      if ( in_toroidal_ ) {
        int vsize = *size;
        treceive_.resize(vsize);

        // synchronize with all operations to finish
        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(toroidal_comm_.size(),
            &generation);

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

    void partition::set_tdata(std::size_t which,
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

    void partition::toroidal_sndright(double *csend,int* mgrid)
    {
//       std::cout << "toroidal_sndright: " << item_ << " -> " << right_pe_
//                 << " (g: " << sndright_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        std::size_t generation = 0;
        std::vector<double> send;

        {
            mutex_type::scoped_lock l(mtx_);

            int vsize = *mgrid;
            sendright_receive_.resize(vsize);
            sndright_future_ = sndright_gate_.get_future(&generation);

            // Send data to the right
            // The sender: send data to the left
            // in a fire and forget fashion
            send.resize(vsize);

            for (std::size_t i=0;i<send.size();i++) {
              send[i] = csend[i];
            }
        }

        set_sendright_data_action set_sendright_data_;
        hpx::apply(set_sendright_data_,
             toroidal_comm_[right_pe_], item_, generation, send);
      } else {
        mutex_type::scoped_lock l(mtx_);
        sndright_gate_.next_generation();
      }
    }

    void partition::toroidal_rcvleft(double *creceive)
    {
//       std::cout << "toroidal_rcvleft: " << item_
//                 << " (g: " << sndright_gate_.generation() << "), "
//                 << in_toroidal_ << std::endl;

      if ( in_toroidal_ ) {
        // Now receive a message from the right
        mutex_type::scoped_lock l(mtx_);
        hpx::future<void> f = sndright_future_;

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        for (std::size_t i=0;i<sendright_receive_.size();i++) {
          creceive[i] = sendright_receive_[i];
        }
//         std::cout << "toroidal_rcvleft: " << item_
//                   << " (g: " << sndright_gate_.generation() << ")"
//                   << std::endl;
      }
    }

    void partition::set_sendright_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        mutex_type::scoped_lock l(mtx_);

//         std::cout << "set_sendright_data: " << item_ << " <- " << which
//                 << " (g: " << sndright_gate_.generation() << ", " << generation << ")"
//                 << std::endl;

        sndright_gate_.synchronize(generation, l, "point::set_sendright_data");
        sendright_receive_ = send;
        sndright_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::toroidal_gather(double *csend, int *tsize,int *tdst)
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
          gather_future_ = gather_gate_.get_future(toroidal_comm_.size(),
              &generation);
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

    void partition::toroidal_gather_receive(double *creceive, int *cdst)
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

    void partition::set_toroidal_gather_data(std::size_t which,
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

    void partition::toroidal_scatter(double *csend, int *tsize,int *tsrc)
    {
      int src = *tsrc;
      if ( src == item_ ) {
        // create a new and-gate object
        std::size_t generation = 0;
        std::vector<double> send;

        {
          mutex_type::scoped_lock l(mtx_);

          int vsize = *tsize;
          toroidal_scatter_receive_.resize(vsize);

          scatter_future_ = scatter_gate_.get_future(&generation);

          // Send data to everyone in toroidal
          // The sender: send data to the left
          // in a fire and forget fashion
          send.resize(vsize);

          for (std::size_t step=0;step<toroidal_comm_.size();step++) {
            for (std::size_t i=0;i<send.size();i++) {
              send[i] = csend[i+step*vsize];
            }
          }
        }

        set_toroidal_scatter_data_action set_toroidal_scatter_data_;
        for (std::size_t i=0;i<toroidal_comm_.size();i++) {
          hpx::apply(set_toroidal_scatter_data_,
              toroidal_comm_[i], item_, generation, send);
        }
      } else {
        scatter_gate_.next_generation();
      }

    }

    void partition::toroidal_scatter_receive(double *creceive, int *cdst)
    {
      if ( in_toroidal_ ) {
        // synchronize with all operations to finish
        mutex_type::scoped_lock l(mtx_);
        hpx::future<void> f = scatter_future_;

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        for (std::size_t i=0;i<toroidal_scatter_receive_.size();i++) {
          creceive[i] = toroidal_scatter_receive_[i];
        }
      }
    }

    void partition::set_toroidal_scatter_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        scatter_gate_.synchronize(generation, l, "point::set_toroidal_scatter_data");
        toroidal_scatter_receive_ = send;
        scatter_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::comm_allreduce(double *in,double *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        comm_allreduce_receive_.resize(vsize);

        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(components_.size(),
            &generation);

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

    void partition::set_comm_allreduce_data(std::size_t which,
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

    void partition::int_comm_allreduce(int *in,int *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        int_comm_allreduce_receive_.resize(vsize);

        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(components_.size(),
            &generation);

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

    void partition::set_int_comm_allreduce_data(std::size_t which,
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

    void partition::int_toroidal_sndright(int *csend,int* mgrid)
    {
      if ( in_toroidal_ ) {

        std::size_t generation = 0;
        std::vector<int> send;

        {
            mutex_type::scoped_lock l(mtx_);

            int vsize = *mgrid;
            int_sendright_receive_.resize(vsize);
            sndright_future_ = sndright_gate_.get_future(&generation);

            // Send data to the right
            // The sender: send data to the left
            // in a fire and forget fashion
            send.resize(vsize);

            for (std::size_t i=0;i<send.size();i++) {
              send[i] = csend[i];
            }
        }

        set_int_sendright_data_action set_int_sendright_data_;
        hpx::apply(set_int_sendright_data_,
             toroidal_comm_[right_pe_], item_, generation, send);
      } else {
        mutex_type::scoped_lock l(mtx_);
        sndright_gate_.next_generation();
      }
    }

    void partition::int_toroidal_rcvleft(int *creceive)
    {
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        mutex_type::scoped_lock l(mtx_);
        hpx::future<void> f = sndright_future_;

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        for (std::size_t i=0;i<int_sendright_receive_.size();i++) {
          creceive[i] = int_sendright_receive_[i];
        }
      }
    }

    void partition::set_int_sendright_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        sndright_gate_.synchronize(generation, l, "point::set_int_sendright_data");
        int_sendright_receive_ = send;
        sndright_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::int_toroidal_sndleft(int *csend,int* mgrid)
    {
      if ( in_toroidal_ ) {
        std::size_t generation = 0;
        std::vector<int> send;

        {
            mutex_type::scoped_lock l(mtx_);

            int vsize = *mgrid;
            int_sendleft_receive_.resize(vsize);

            // create a new and-gate object
            sndleft_future_ = sndleft_gate_.get_future(&generation);

            // Send data to the left
            // The sender: send data to the left
            // in a fire and forget fashion
            send.resize(vsize);

            for (std::size_t i=0;i<send.size();i++) {
              send[i] = csend[i];
            }
        }

        set_int_sendleft_data_action set_int_sendleft_data_;
        hpx::apply(set_int_sendleft_data_, toroidal_comm_[left_pe_], item_, 
            generation, send);
      } else {
        mutex_type::scoped_lock l(mtx_);
        sndleft_gate_.next_generation();
      }
    }

    void partition::int_toroidal_rcvright(int *creceive)
    {
      if ( in_toroidal_ ) {
        // Now receive a message from the right
        mutex_type::scoped_lock l(mtx_);
        hpx::future<void> f = sndleft_future_;

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        for (std::size_t i=0;i<int_sendleft_receive_.size();i++) {
          creceive[i] = int_sendleft_receive_[i];
        }
      }
    }

    void partition::set_int_sendleft_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        sndleft_gate_.synchronize(generation, l, "point::set_int_sendleft_data");
        int_sendleft_receive_ = send;
        sndleft_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::set_toroidal_cmm(int *send,int*length)
    {
      t_comm_.resize(*length);
      for (std::size_t i=0;i<*length;i++) {
        t_comm_[i] = send[i];
      }
    }

    void partition::set_partd_cmm(int *send,int*length)
    {
      p_comm_.resize(*length);
      for (std::size_t i=0;i<*length;i++) {
        p_comm_[i] = send[i];
      }
    }

    void partition::int_comm_allgather(int *in,int *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        int_comm_allgather_receive_.resize(components_.size()*vsize);

        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(components_.size(),
            &generation);

        std::vector<int> send;
        send.resize(vsize);

        for (std::size_t i=0;i<send.size();i++) {
          send[i] = in[i];
        }

        set_int_comm_allgather_data_action set_allgather;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_allgather, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<int_comm_allgather_receive_.size();i++) {
          out[i] = int_comm_allgather_receive_[i];
        }
    }

    void partition::set_int_comm_allgather_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data)
    {
        allgather_gate_.synchronize(generation, "point::set_int_comm_allgather_data");

        {
            mutex_type::scoped_lock l(mtx_);
            std::size_t vsize = int_comm_allgather_receive_.size();
            for (std::size_t i=0;i<vsize;i++)
                int_comm_allgather_receive_[which*vsize + i] = data[i];
        }

        allgather_gate_.set(which);         // trigger corresponding and-gate input
    }

}}

