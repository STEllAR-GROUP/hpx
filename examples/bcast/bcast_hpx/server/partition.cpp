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

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

int floatcmp(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}

void worker(boost::uint64_t delay, volatile double * d)
{
    for (boost::uint64_t i = 0; i < delay; ++i)
        *d += 1. / (2. * i + 1.);
}

///////////////////////////////////////////////////////////////////////////////
namespace bcast { namespace server
{
    void partition::loop_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components,
                      std::size_t sendbuf,std::size_t sendbuf2,std::size_t delay)
    {
      item_ = mype;
      components_ = components;

      int bcast_int_array_size = sendbuf;
      int bcast_double_array_size = sendbuf;
      int *bcast_int_array; 
      double *bcast_double_array;

      bcast_int_array = (int *) malloc(sizeof(int)*bcast_int_array_size);
      bcast_double_array = (double *) malloc(sizeof(double)*bcast_double_array_size);

      if ( item_ == 0 ) {
        for (int i=0;i<bcast_int_array_size;i++) {
          bcast_int_array[i] = i;
        }
        for (int i=0;i<bcast_double_array_size;i++) {
          bcast_double_array[i] = 1.0*i;
        }
      }

      broadcast_parameters(bcast_int_array,bcast_double_array,bcast_int_array_size,bcast_double_array_size);

      // Verify broadcast worked
      for (int i=0;i<bcast_int_array_size;i++) {
        if ( bcast_int_array[i] != i ) {
          std::cout << " Problem in integer broadcast! " << bcast_int_array[i] << " i " << i << std::endl;
        }
      }
      for (int i=0;i<bcast_double_array_size;i++) {
        if ( floatcmp(bcast_double_array[i],1.0*i)!= 1) {
          std::cout << " Problem in double broadcast! " << bcast_double_array[i] << " i " << i << std::endl;
        }
      }

      free(bcast_int_array);
      free(bcast_double_array);

      // add some busy work if requested
      volatile double d = 0;
      worker(delay, &d);

      double *send_array;
      double *receive_array;
      send_array = (double *) malloc(sizeof(double)*sendbuf2);
      receive_array = (double *) malloc(sizeof(double)*sendbuf2);

      int dest = item_ + 1; 
      if ( dest >= numberpe ) dest = 0;

      for (int i=0;i<sendbuf2;i++) {
        send_array[i] = item_*1000.0 + i; 
      }

      // send receive in a periodic fashion
      toroidal_sndrecv(send_array,sendbuf2,receive_array,sendbuf2,dest);

      // Validate the send/receive
      int origin = item_-1;
      if ( origin < 0 ) origin = numberpe-1;
      for (int i=0;i<sendbuf2;i++) {
        if ( floatcmp(receive_array[i],origin*1000.0+i) != 1 ) {
          std::cout << " Problem in send receive! " << receive_array[i] << " mype " << mype << " i " << i << std::endl;
        }
      }
      
    }

    void partition::broadcast_parameters(int *integer_params,double *real_params,
                             int nint,int nreal)
    {
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











    void partition::toroidal_sndrecv(double *csend,int send_size,double *creceive,int receive_size,int dest)
    {
      std::size_t generation = 0;
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
      hpx::apply(set_sndrecv_data_, components_[dest], item_,
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

}}

