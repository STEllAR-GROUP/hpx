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
            void FNAME(loop)(void* opaque_ptr_to_class, int *,int *);
            void FNAME(sndrecv_toroidal_cmm) (void* pfoo,double *send, int *send_size,
                                               double *receive,int *receive_size,int *dest,
                                               int* repeats) {
                    hpx::util::high_resolution_timer computetime;
                    sendrecv::server::partition *ptr_to_class = *static_cast<sendrecv::server::partition**>(pfoo);
                    for (std::size_t i=0;i<*repeats;i++) {
                      ptr_to_class->toroidal_sndrecv(send,send_size,receive,receive_size,dest);
                    }
                    double ctime = computetime.elapsed();
                    std::cout << " Time " << ctime << std::endl;
                    return; };
}

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
namespace sendrecv { namespace server
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
      hpx::apply(set_sndrecv_data_, components_[*dest], item_,
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

