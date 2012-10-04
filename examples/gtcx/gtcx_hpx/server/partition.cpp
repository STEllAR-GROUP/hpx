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
            void FNAME(set_partd_cmm) (void* pfoo,int *send, int* length,int *myrank_partd) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->set_partd_cmm(send,length,myrank_partd);
                    return; };
            void FNAME(send_cmm) (void* pfoo,double *send, int* length,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->p2p_send(send,length,dest);
                    return; };
            void FNAME(receive_cmm) (void* pfoo,double *receive, int* length,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->p2p_receive(receive,length,dest);
                    return; };
            void FNAME(set_toroidal_cmm) (void* pfoo,int *send, int* length,int *myrank_toroidal) {
                    // Cast to gtcx::server::point.  If the opaque pointer isn't a pointer to an object
                    // derived from point, then the world will end.
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->set_toroidal_cmm(send,length,myrank_toroidal);
                    return; };
            void FNAME(loop)(void* opaque_ptr_to_class, int *,int *);
            void FNAME(sndrecv_toroidal_cmm) (void* pfoo,double *send, int *send_size,
                                               double *receive,int *receive_size,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_sndrecv(send,send_size,receive,receive_size,dest);
                    return; };
            void FNAME(int_sndrecv_toroidal_cmm) (void* pfoo,int *send, int *send_size,
                                               int *receive,int *receive_size,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->int_toroidal_sndrecv(send,send_size,receive,receive_size,dest);
                    return; };
            void FNAME(partd_allreduce_cmm) (void* pfoo,double *dnitmp,double *densityi,
                                             int* mgrid, int *mzetap1) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->partd_allreduce(dnitmp,densityi,mgrid,mzetap1);
                    return; };
            void FNAME(partd_allgather_cmm) (void* pfoo,double *in,double *out, int* size) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->partd_allgather(in,out,size);
                    return; };
            void FNAME(toroidal_allreduce_cmm) (void* pfoo,double *input,double *output,
                                             int* size) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_allreduce(input,output,size);
                    return; };
            void FNAME(toroidal_reduce_cmm) (void* pfoo,double *input,double *output,
                                             int* size,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->toroidal_reduce(input,output,size,dest);
                    return; };
            void FNAME(comm_reduce_cmm) (void* pfoo,double *input,double *output,
                                             int* size,int *dest) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->comm_reduce(input,output,size,dest);
                    return; };
            void FNAME(broadcast_int_cmm) (void* pfoo,
                     int *integer_params, int *n_integers) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->broadcast_int_parameters(integer_params, n_integers);
                    return; };
            void FNAME(broadcast_real_cmm) (void* pfoo, double *real_params,int *n_reals) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->broadcast_real_parameters(real_params, n_reals);
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
            void FNAME(ntoroidal_gather_cmm) (void* pfoo,double *input,
                                             int* size,double *output, int* dst) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->ntoroidal_gather(input,size,output,dst);
                    return; };
            void FNAME(complex_ntoroidal_gather_cmm) (void* pfoo,std::complex<double> *input,
                                             int* size,std::complex<double> *output, int* dst) {
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->complex_ntoroidal_gather(input,size,output,dst);
                    return; };
            void FNAME(ntoroidal_scatter_cmm) (void* pfoo,double *input,int *size,double *output,int * src){
                    gtcx::server::partition *ptr_to_class = *static_cast<gtcx::server::partition**>(pfoo);
                    ptr_to_class->ntoroidal_scatter(input,size,output,src);
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
        BOOST_ASSERT(intparams_.size() == nint);
        for (std::size_t i=0;i<intparams_.size();i++) {
          integer_params[i] = intparams_[i];
        }
        BOOST_ASSERT(realparams_.size() == nreal);
        for (std::size_t i=0;i<realparams_.size();i++) {
          real_params[i] = realparams_[i];
        }
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = broadcast_gate_.next_generation();

        std::vector<int> intparams(integer_params, integer_params+nint);
        std::vector<double> realparams(real_params, real_params+nreal);

        // eliminate item 0's (the sender's) gid
        std::vector<hpx::naming::id_type> all_but_root(components_.size()-1);
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
      int vsize = (*mgrid)*(*mzetap1);
      dnireceive_.resize(vsize);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f = allreduce_gate_.get_future(p_comm_.size(),
          &generation);

      std::vector<double> dnisend(dnitmp, dnitmp+vsize);

      set_data_action set_data_;
      for (std::size_t i=0;i<p_comm_.size();i++) {
        hpx::apply(set_data_, components_[p_comm_[i]], myrank_partd_, generation, dnisend);
      }

      // possibly do other stuff while the allgather is going on...
      f.get();

      mutex_type::scoped_lock l(mtx_);
      BOOST_ASSERT(dnireceive_.size() == vsize);
      for (std::size_t i=0;i<dnireceive_.size();i++) {
        densityi[i] = dnireceive_[i];
      }
    }

    void partition::set_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(dnireceive_.size() == data.size());
            for (std::size_t i=0;i<dnireceive_.size();i++)
                dnireceive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

















    void partition::partd_allgather(double *in,double *out, int* size)
    {
      int vsize = *size;
      partd_allgather_.resize(p_comm_.size()*vsize);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f = partd_allgather_gate_.get_future(p_comm_.size(),
          &generation);

      std::vector<double> send(in, in+vsize);

      set_partd_allgather_data_action set_partd_allgather_data_;
      for (std::size_t i=0;i<p_comm_.size();i++) {
        hpx::apply(set_partd_allgather_data_, components_[p_comm_[i]], myrank_partd_, generation, send);
      }

      // possibly do other stuff while the allgather is going on...
      f.get();

      mutex_type::scoped_lock l(mtx_);
      BOOST_ASSERT(partd_allgather_.size() == p_comm_.size()*vsize);
      for (std::size_t i=0;i<partd_allgather_.size();i++) {
        out[i] = partd_allgather_[i];
      }
    }

    void partition::set_partd_allgather_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        partd_allgather_gate_.synchronize(generation, "point::set_partd_allgather_data");

        {
            mutex_type::scoped_lock l(mtx_);
            std::size_t vsize = data.size();
            BOOST_ASSERT(partd_allgather_.size() == p_comm_.size()*vsize);
            for (std::size_t i=0;i<vsize;i++)
                partd_allgather_[which*vsize + i] = data[i];
        }

        partd_allgather_gate_.set(which);         // trigger corresponding and-gate input
    }

  














    void partition::broadcast_int_parameters(int *integer_params, int *n_integers)
    {
      int nint = *n_integers;

      if ( item_ != 0 ) {
        // synchronize with all operations to finish
        hpx::future<void> f = broadcast_gate_.get_future(1);

        f.get();

        // Copy the parameters to the fortran arrays
        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(intparams_.size() == nint);
        for (std::size_t i=0;i<intparams_.size();i++) {
          integer_params[i] = intparams_[i];
        }
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = broadcast_gate_.next_generation();

        std::vector<int> intparams(integer_params, integer_params+nint);

        // eliminate item 0's (the sender's) gid
        std::vector<hpx::naming::id_type> all_but_root(components_.size()-1);
        for (std::size_t i=0;i<all_but_root.size();i++) {
          all_but_root[i] = components_[i+1];
        }

        set_int_params_action set_int_params_;
        for (std::size_t i=0;i<all_but_root.size();i++) {
          hpx::apply(set_int_params_, all_but_root[i], item_, generation,
                     intparams);
        }
      }
    }

    void partition::set_int_params(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& intparams)
    {
        broadcast_gate_.synchronize(generation, "point::set_int_params");

        {
            mutex_type::scoped_lock l(mtx_);
            intparams_ = intparams;
        }

        // which is always zero in this case
        broadcast_gate_.set(which);         // trigger corresponding and-gate input
    }


















    void partition::broadcast_real_parameters(double *real_params,int *n_reals)
    {
      int nreal = *n_reals;

      if ( item_ != 0 ) {
        // synchronize with all operations to finish
        hpx::future<void> f = broadcast_gate_.get_future(1);

        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(realparams_.size() == nreal);
        for (std::size_t i=0;i<realparams_.size();i++) {
          real_params[i] = realparams_[i];
        }
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = broadcast_gate_.next_generation();

        std::vector<double> realparams(real_params, real_params+nreal);

        // eliminate item 0's (the sender's) gid
        std::vector<hpx::naming::id_type> all_but_root(components_.size()-1);
        for (std::size_t i=0;i<all_but_root.size();i++) {
          all_but_root[i] = components_[i+1];
        }

        set_real_params_action set_real_params_;
        for (std::size_t i=0;i<all_but_root.size();i++) {
          hpx::apply(set_real_params_, all_but_root[i], item_, generation,
                     realparams);
        }
      }
    }

    void partition::set_real_params(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& realparams)
    {
        broadcast_gate_.synchronize(generation, "point::set_real_params");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(realparams_.size() == realparams.size());
            realparams_ = realparams;
        }

        // which is always zero in this case
        broadcast_gate_.set(which);         // trigger corresponding and-gate input
    }














    void partition::toroidal_sndrecv(double *csend,int* csend_size,double *creceive,int *creceive_size,int* dest)
    {
      std::size_t generation = 0;
      int send_size = *csend_size;
      int receive_size = *creceive_size;
      hpx::future<void> f;

      {
        mutex_type::scoped_lock l(mtx_);
        sndrecv_.resize(receive_size);
        f = sndrecv_gate_.get_future(&generation);
      }

      // The sender: send data to the left
      // in a fire and forget fashion
      std::vector<double> send(csend, csend+send_size);

      // send message to the left
      set_sndrecv_data_action set_sndrecv_data_;
      hpx::apply(set_sndrecv_data_, components_[t_comm_[*dest]], item_,
          generation, send);

      //std::cerr << "toroidal_sndrecv(" << item_ << "): " 
      //          << "g(" << generation << "), " 
      //          << "s(" << send_size << "), r(" << receive_size << ")" 
      //          << std::endl;

      // Now receive a message from the right
      f.get();

      mutex_type::scoped_lock l(mtx_);
      BOOST_ASSERT(sndrecv_.size() == receive_size);
      for (std::size_t i=0;i<sndrecv_.size();i++) {
        creceive[i] = sndrecv_[i];
      }
    }

    void partition::set_sndrecv_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        sndrecv_gate_.synchronize(generation, l, "point::set_sndrecv_data");

        //std::cerr << "set_sndrecv_data(" << item_ << "," << which << "): " 
        //        << "g(" << generation << "), " 
        //        << "s(" << send.size() << "), r(" << sndrecv_.size() << ")"
        //        << std::endl;

        BOOST_ASSERT(sndrecv_.size() == send.size());
        sndrecv_ = send;
        sndrecv_gate_.set();         // trigger corresponding and-gate input
    }









    void partition::int_toroidal_sndrecv(int *csend,int* csend_size,int *creceive,int *creceive_size,int* dest)
    {
      std::size_t generation = 0;
      int send_size = *csend_size;
      int receive_size = *creceive_size;
      hpx::future<void> f;

      {
        mutex_type::scoped_lock l(mtx_);
        int_sndrecv_.resize(receive_size);
        f = sndrecv_gate_.get_future(&generation);
      }

      // The sender: send data to the left
      // in a fire and forget fashion
      std::vector<int> send(csend, csend+send_size);

      // send message to the left
      set_int_sndrecv_data_action set_int_sndrecv_data_;
      hpx::apply(set_int_sndrecv_data_, components_[t_comm_[*dest]], item_,
          generation, send);

      //std::cerr << "int_toroidal_sndrecv(" << item_ << "): " 
      //          << "g(" << generation << "), " 
      //          << "s(" << send_size << "), r(" << receive_size << ")" 
      //          << std::endl;

      // Now receive a message from the right
      f.get();

      {
        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(int_sndrecv_.size() == receive_size);
        for (std::size_t i=0;i<int_sndrecv_.size();i++) {
          creceive[i] = int_sndrecv_[i];
        }
      }
    }

    void partition::set_int_sndrecv_data(std::size_t which,
                           std::size_t generation,
                           std::vector<int> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        sndrecv_gate_.synchronize(generation, l, "point::set_int_sndrecv_data");

        //std::cerr << "set_int_sndrecv_data(" << item_ << "," << which << "): " 
        //        << "g(" << generation << "), " 
        //        << "s(" << send.size() << "), r(" << int_sndrecv_.size() << ")"
        //        << std::endl;

        BOOST_ASSERT(int_sndrecv_.size() == send.size());
        int_sndrecv_ = send;
        sndrecv_gate_.set();         // trigger corresponding and-gate input
    }











    void partition::p2p_receive(double *creceive, int *csize,int *tdst)
    {
      int dst = *tdst;
      int vsize = *csize;
      if ( item_ != dst ) {
        std::cout << " Problem:  p2p_receive needs to be called only by the receiver. " << item_ << " " << dst << std::endl;
      }

      p2p_sendreceive_future_.get();

      mutex_type::scoped_lock l(mtx_);
      if ( p2p_sendreceive_.size() != vsize ) {
        std::cout << " Problem:  receive size doesn't match send size in p2p_receive " << std::endl;
      }

      for (std::size_t i=0;i<p2p_sendreceive_.size();i++) {
        creceive[i] = p2p_sendreceive_[i];
      }
      
    }

    void partition::p2p_send(double *csend, int *csize,int *tdst)
    {
      int vsize = *csize;
      int dst = *tdst;

      // create a new and-gate object
      std::size_t generation = 0;
      {
        mutex_type::scoped_lock l(mtx_);
        p2p_sendreceive_.resize(vsize);
        p2p_sendreceive_future_ = p2p_sendreceive_gate_.get_future(&generation);
      }

      // Send data to dst
      // The sender: send data to the left
      // in a fire and forget fashion
      std::vector<double> send(csend, csend+vsize);

      set_p2p_sendreceive_data_action set_p2p_sendreceive_data_;
      hpx::apply(set_p2p_sendreceive_data_,
           components_[dst], item_, generation, send);
    }


    void partition::set_p2p_sendreceive_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        p2p_sendreceive_gate_.synchronize(generation, l, "point::set_p2p_sendreceive_data");
        BOOST_ASSERT(p2p_sendreceive_.size() == send.size());
        p2p_sendreceive_ = send;
        p2p_sendreceive_gate_.set();
    }






    void partition::toroidal_reduce(double *input,double *output, int* size,int *tdest)
    {
      int dest = *tdest;
      int vsize = *size;
      toroidal_reduce_.resize(vsize);
      std::fill( toroidal_reduce_.begin(),toroidal_reduce_.end(),0.0);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f;
      if ( myrank_toroidal_ == dest ) {
        f = toroidal_reduce_gate_.get_future(t_comm_.size(), &generation);
      } else {
        generation = toroidal_reduce_gate_.next_generation();
      }

      std::vector<double> send(input, input+vsize);

      set_toroidal_reduce_data_action set_toroidal_reduce_data_;
      hpx::apply(set_toroidal_reduce_data_, 
             components_[t_comm_[dest]], myrank_toroidal_, generation, send);

      if ( myrank_toroidal_ == dest ) {
        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(toroidal_reduce_.size() == vsize);
        for (std::size_t i=0;i<toroidal_reduce_.size();i++) {
          output[i] = toroidal_reduce_[i];
        }
      }
    }

    void partition::set_toroidal_reduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        toroidal_reduce_gate_.synchronize(generation, "point::set_toroidal_reduce_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(toroidal_reduce_.size() == data.size());
            for (std::size_t i=0;i<toroidal_reduce_.size();i++)
                toroidal_reduce_[i] += data[i];
        }

        toroidal_reduce_gate_.set(which);         // trigger corresponding and-gate input
    }


















    void partition::comm_reduce(double *input,double *output, int* size,int *tdest)
    {
      int dest = *tdest;
      int vsize = *size;
      comm_reduce_.resize(vsize);
      std::fill( comm_reduce_.begin(),comm_reduce_.end(),0.0);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f;
      if ( item_ == dest ) {
        f = comm_reduce_gate_.get_future(components_.size(),
            &generation);
      } else {
        generation = comm_reduce_gate_.next_generation();
      }

      std::vector<double> send(input, input+vsize);

      set_comm_reduce_data_action set_comm_reduce_data_;
      hpx::apply(set_comm_reduce_data_, 
             components_[dest], item_, generation, send);

      if ( item_ == dest ) {
        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(comm_reduce_.size() == vsize);
        for (std::size_t i=0;i<comm_reduce_.size();i++) {
          output[i] = comm_reduce_[i];
        }
      }
    }

    void partition::set_comm_reduce_data(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        comm_reduce_gate_.synchronize(generation, "point::set_comm_reduce_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(comm_reduce_.size() == data.size());
            for (std::size_t i=0;i<comm_reduce_.size();i++)
                comm_reduce_[i] += data[i];
        }

        comm_reduce_gate_.set(which);         // trigger corresponding and-gate input
    }















    void partition::toroidal_allreduce(double *input,double *output, int* size)
    {
      int vsize = *size;
      treceive_.resize(vsize);
      std::fill( treceive_.begin(),treceive_.end(),0.0);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f = toroidal_allreduce_gate_.get_future(t_comm_.size(),
            &generation);

      std::vector<double> send(input, input+vsize);

      set_tdata_action set_tdata_;
      for (std::size_t i=0;i<t_comm_.size();i++) {
        hpx::apply(set_tdata_, components_[t_comm_[i]], myrank_toroidal_, generation, send);
      }

      // possibly do other stuff while the allgather is going on...
      f.get();

      mutex_type::scoped_lock l(mtx_);
      BOOST_ASSERT(treceive_.size() == vsize);
      for (std::size_t i=0;i<treceive_.size();i++) {
        output[i] = treceive_[i];
      }
    }

    void partition::set_tdata(std::size_t which,
                std::size_t generation, std::vector<double> const& data)
    {
        toroidal_allreduce_gate_.synchronize(generation, "point::set_tdata");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(treceive_.size() == data.size());
            for (std::size_t i=0;i<treceive_.size();i++)
                treceive_[i] += data[i];
        }

        toroidal_allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void partition::ntoroidal_gather(double *csend, int *csize,double *creceive,int *tdst)
    {
      int vsize = *csize;
      int dst = *tdst;

      hpx::future<void> f;

      // create a new and-gate object
      std::size_t generation = 0;
      if ( myrank_toroidal_ == dst ) {
        ntoroidal_gather_receive_.resize(t_comm_.size()*vsize);
        f = gather_gate_.get_future(t_comm_.size(), &generation);
      }
      else {
        generation = gather_gate_.next_generation();
      }

      // Send data to dst
      // The sender: send data to the left
      // in a fire and forget fashion
      std::vector<double> send(csend, csend+vsize);

      set_ntoroidal_gather_data_action set_ntoroidal_gather_data_;
      hpx::apply(set_ntoroidal_gather_data_,
           components_[t_comm_[dst]], myrank_toroidal_, generation, send);

      if ( myrank_toroidal_ == dst ) {
        // synchronize with all operations to finish
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(ntoroidal_gather_receive_.size() == t_comm_.size()*vsize);
        for (std::size_t i=0;i<ntoroidal_gather_receive_.size();i++) {
          creceive[i] = ntoroidal_gather_receive_[i];
        }
      }
    }

    void partition::set_ntoroidal_gather_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        gather_gate_.synchronize(generation, "point::set_ntoroidal_gather_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(ntoroidal_gather_receive_.size() == send.size()*t_comm_.size());
            BOOST_ASSERT(which < t_comm_.size());
            for (std::size_t i=0;i<send.size();i++)
                ntoroidal_gather_receive_[which*send.size()+i] = send[i];
        }

        gather_gate_.set(which);         // trigger corresponding and-gate input
    }

    void partition::complex_ntoroidal_gather(std::complex<double> *csend, int *csize,std::complex<double> *creceive,int *tdst)
    {
      int vsize = *csize;
      int dst = *tdst;

      hpx::future<void> f;

      // create a new and-gate object
      std::size_t generation = 0;
      if ( myrank_toroidal_ == dst ) {
        complex_ntoroidal_gather_receive_.resize(t_comm_.size()*vsize);
        f = gather_gate_.get_future(t_comm_.size(), &generation);
      }
      else {
        generation = gather_gate_.next_generation();
      }

      // Send data to dst
      // The sender: send data to the left
      // in a fire and forget fashion
      std::vector<std::complex<double> > send(csend, csend+vsize);

      set_complex_ntoroidal_gather_data_action set_complex_ntoroidal_gather_data_;
      hpx::apply(set_complex_ntoroidal_gather_data_,
           components_[t_comm_[dst]], myrank_toroidal_, generation, send);

      if ( myrank_toroidal_ == dst ) {
        // synchronize with all operations to finish
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(complex_ntoroidal_gather_receive_.size() == t_comm_.size()*vsize);
        for (std::size_t i=0;i<complex_ntoroidal_gather_receive_.size();i++) {
          creceive[i] = complex_ntoroidal_gather_receive_[i];
        }
      }
    }

    void partition::set_complex_ntoroidal_gather_data(std::size_t which,
                           std::size_t generation,
                           std::vector<std::complex<double> > const& send)
    {
        gather_gate_.synchronize(generation, "point::set_complex_ntoroidal_gather_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(complex_ntoroidal_gather_receive_.size() == send.size()*t_comm_.size());
            BOOST_ASSERT(which < t_comm_.size());
            for (std::size_t i=0;i<send.size();i++) {
              complex_ntoroidal_gather_receive_[which*send.size()+i] = send[i];
            }
        }

        gather_gate_.set(which);         // trigger corresponding and-gate input
    }

    void partition::ntoroidal_scatter(double *csend, int *csize,double *creceive,int *tsrc)
    {
      int src = *tsrc;
      hpx::future<void> f;

      int vsize = *csize;
      ntoroidal_scatter_receive_.resize(vsize);

      std::size_t generation = 0;
      f = scatter_gate_.get_future(&generation);

      if ( t_comm_[src] == item_ ) {
        mutex_type::scoped_lock l(mtx_);

        // Send data to everyone in toroidal
        // The sender: send data to the left
        // in a fire and forget fashion
        std::vector<double> send(vsize);

        set_ntoroidal_scatter_data_action set_ntoroidal_scatter_data_;
        for (std::size_t i=0;i<t_comm_.size();i++) {
          for (std::size_t j=0;j<send.size();j++) {
            send[j] = csend[j+i*vsize];
          }
          hpx::apply(set_ntoroidal_scatter_data_,
              components_[t_comm_[i]], myrank_toroidal_, generation, send);
        }
      }

      // synchronize with all operations to finish
      f.get();

      mutex_type::scoped_lock l(mtx_);
      BOOST_ASSERT(ntoroidal_scatter_receive_.size() == vsize);
      for (std::size_t i=0;i<ntoroidal_scatter_receive_.size();i++) {
        creceive[i] = ntoroidal_scatter_receive_[i];
      }
    }

    void partition::set_ntoroidal_scatter_data(std::size_t which,
                           std::size_t generation,
                           std::vector<double> const& send)
    {
        mutex_type::scoped_lock l(mtx_);
        scatter_gate_.synchronize(generation, l, "point::set_ntoroidal_scatter_data");
        BOOST_ASSERT(ntoroidal_scatter_receive_.size() == send.size());
        ntoroidal_scatter_receive_ = send;
        scatter_gate_.set();         // trigger corresponding and-gate input
    }

    void partition::comm_allreduce(double *in,double *out, int* msize)
    {
        // synchronize with all operations to finish
        int vsize = *msize;
        comm_allreduce_receive_.resize(vsize);
        std::fill( comm_allreduce_receive_.begin(),comm_allreduce_receive_.end(),0.0);

        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(components_.size(),
            &generation);

        std::vector<double> send(in, in+vsize);

        set_comm_allreduce_data_action set_data_;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_data_, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(comm_allreduce_receive_.size() == vsize);
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
            BOOST_ASSERT(comm_allreduce_receive_.size() == data.size());
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
        std::fill( int_comm_allreduce_receive_.begin(),int_comm_allreduce_receive_.end(),0);

        std::size_t generation = 0;
        hpx::future<void> f = allreduce_gate_.get_future(components_.size(),
            &generation);

        std::vector<int> send(in, in+vsize);

        set_int_comm_allreduce_data_action set_data_;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_data_, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(int_comm_allreduce_receive_.size() == vsize);
        for (std::size_t i=0;i<int_comm_allreduce_receive_.size();i++) {
          out[i] = int_comm_allreduce_receive_[i];
        }
    }

    void partition::set_int_comm_allreduce_data(std::size_t which,
                std::size_t generation, std::vector<int> const& data)
    {
        allreduce_gate_.synchronize(generation, "point::set_int_comm_allreduce_data");

        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(int_comm_allreduce_receive_.size() == data.size());
            for (std::size_t i=0;i<int_comm_allreduce_receive_.size();i++)
                int_comm_allreduce_receive_[i] += data[i];
        }

        allreduce_gate_.set(which);         // trigger corresponding and-gate input
    }

    void partition::set_toroidal_cmm(int *send,int*length,int *myrank_toroidal)
    {
      myrank_toroidal_ = *myrank_toroidal;
      t_comm_.resize(*length);
      for (std::size_t i=0;i<*length;i++) {
        t_comm_[i] = send[i];
      }
    }

    void partition::set_partd_cmm(int *send,int*length,int *myrank_partd)
    {
      myrank_partd_ = *myrank_partd;
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

        std::vector<int> send(in, in+vsize);

        set_int_comm_allgather_data_action set_allgather;
        for (std::size_t i=0;i<components_.size();i++) {
          hpx::apply(set_allgather, components_[i], item_, generation, send);
        }

        // possibly do other stuff while the allgather is going on...
        f.get();

        mutex_type::scoped_lock l(mtx_);
        BOOST_ASSERT(int_comm_allgather_receive_.size() == vsize);
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
            std::size_t vsize = data.size();
            BOOST_ASSERT(int_comm_allgather_receive_.size() == data.size()*components_.size());
            BOOST_ASSERT(which < components_.size());
            for (std::size_t i=0;i<vsize;i++)
                int_comm_allgather_receive_[which*vsize + i] = data[i];
        }

        allgather_gate_.set(which);         // trigger corresponding and-gate input
    }

}}

