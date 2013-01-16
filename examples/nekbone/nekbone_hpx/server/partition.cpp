//  Copyright (c) 2012 Matthew Anderson
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
  void FNAME(nekproxy)(void* opaque_ptr_to_class, int *,int *);
  void FNAME(broadcast_int_cmm)(void* pfoo,int *integer_par) {
    nekbone::server::partition *ptr_to_class = *static_cast<nekbone::server::partition**>(pfoo);
    ptr_to_class->broadcast_int_parameters(integer_par);
  }
  void FNAME(double_mpi_allreduce_cmm)(void* pfoo,double *x, double *w,int *n,int *ierr) {
    nekbone::server::partition *ptr_to_class = *static_cast<nekbone::server::partition**>(pfoo);
    ptr_to_class->double_mpi_allreduce(x,w,n,ierr);
  }
}

///////////////////////////////////////////////////////////////////////////////
inline void set_description(char const* test_name)
{
    hpx::threads::set_thread_description(hpx::threads::get_self_id(), test_name);
}

///////////////////////////////////////////////////////////////////////////////
namespace nekbone { namespace server
{
    void partition::loop_wrapper(std::size_t numberpe,std::size_t mype,
                      std::vector<hpx::naming::id_type> const& components)
    {
      item_ = mype;
      components_ = components;
 
      // eliminate item 0's (the sender's) gid
      all_but_root_.resize(components_.size()-1);
      for (std::size_t i=0;i<all_but_root_.size();i++) {
        all_but_root_[i] = components_[i+1];
      }

      int t2 = static_cast<int>(numberpe);
      int t1 = static_cast<int>(mype);

      FNAME(nekproxy)(static_cast<void*>(this), &t1,&t2);

    }

    void partition::broadcast_int_parameters(int *integer_par)
    {
      if ( item_ != 0 ) {
        // synchronize with all operations to finish
        hpx::future<void> f = broadcast_gate_.get_future(1);

        f.get();
 
        *integer_par = intparams_;
      } else {
        // The sender:  broadcast the parameters to the other components
        // in a fire and forget fashion
        std::size_t generation = broadcast_gate_.next_generation(); 

        int intparams = *integer_par;

        set_int_params_action set_int_params_;
        for (std::size_t i=0;i<all_but_root_.size();i++) {
          hpx::apply(set_int_params_, all_but_root_[i], item_, generation,
                     intparams);
        }
      }
    }

    void partition::set_int_params(std::size_t which,
                           std::size_t generation,
                           int intparams)
    {
        broadcast_gate_.synchronize(generation, "point::set_int_params");

        {
            mutex_type::scoped_lock l(mtx_);
            intparams_ = intparams;
        }

        // which is always zero in this case
        broadcast_gate_.set(which);         // trigger corresponding and-gate input
    }

    void partition::double_mpi_allreduce(double *x, double *w,int *n,int *ierr)
    {
      int vsize = *n;
      mpi_allreduce_data_.resize(vsize);
      std::fill(mpi_allreduce_data_.begin(),mpi_allreduce_data_.end(),0.0);

      // synchronize with all operations to finish
      std::size_t generation = 0;
      hpx::future<void> f = double_allreduce_gate_.get_future(components_.size(),
          &generation);

      std::vector<double> send(x, x+vsize);

      set_double_mpi_allreduce_action set_data_;
      for (std::size_t i=0;i<components_.size();i++) {
        hpx::apply(set_data_, components_[i], item_, generation, send);
      }

      // possibly do other stuff while the allgather is going on...
      f.get();
      
      mutex_type::scoped_lock l(mtx_);
      for (std::size_t i=0;i<vsize;i++) {
        x[i] = mpi_allreduce_data_[i];
      }
    }

    void partition::set_double_mpi_allreduce(std::size_t which,std::size_t generation,
                                             std::vector<double> const& data)
    {
      double_allreduce_gate_.synchronize(generation,"partition::set_double_mpi_allreduce");
      {
        mutex_type::scoped_lock l(mtx_);
        for (std::size_t i=0;i<mpi_allreduce_data_.size();i++)
          mpi_allreduce_data_[i] += data[i];
      }
      double_allreduce_gate_.set(which); // trigger corresponding and-gate input
    }

}}

