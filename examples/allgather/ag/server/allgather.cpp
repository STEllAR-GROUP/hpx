//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/unwrapped.hpp>

#include "allgather.hpp"

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/format.hpp>
#include <boost/thread/locks.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
{
    void allgather::init(std::size_t item,std::size_t np)
    {
      //boost::lock_guard<hpx::util::spinlock> l(mtx_);
      item_ = item;
      value_ = item*3.14159;
    }

    void allgather::compute(std::vector<hpx::naming::id_type> const& point_components)
    {
     // boost::lock_guard<hpx::util::spinlock> l(mtx_);
      typedef std::vector<hpx::lcos::future< double > > lazy_results_type;
      lazy_results_type lazy_results;

      server::allgather::get_item_action get_item_;
      for (std::size_t i=0;i<point_components.size();i++)
      {
        lazy_results.push_back( hpx::async(get_item_, point_components[i]) );
      }

      n_ = hpx::util::unwrapped(lazy_results);
    }

    double allgather::get_item() const
    {
      //std::cout << " Get_item " << item_ << std::endl;
      //boost::lock_guard<hpx::util::spinlock> l(mtx_);
      return value_;
    }

    void allgather::print()
    {
      std::cout << " location: " << item_ << " n size : " << n_.size() << std::endl;
      for ( std::size_t i=0;i<n_.size();i++) {
        std::cout << "     n value: " << n_[i] << std::endl;
      }
    }

}}

