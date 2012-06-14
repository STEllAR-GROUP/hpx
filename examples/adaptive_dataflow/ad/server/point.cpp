//  Copyright (c) 2011 Matthew Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/lcos/future_wait.hpp>

#include "../stubs/point.hpp"
#include "point.hpp"

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace ad { namespace server
{
    void point::init(std::size_t item,std::size_t np)
    {
      //hpx::util::spinlock::scoped_lock l(mtx_);
      item_ = item;  
      active_ = true;

      std::size_t left,right;

      if ( item > 0 ) left = item-1;
      else left = np-1;

      if ( item < np-1 ) right = item+1;
      else right = 0;

      neighbors_.push_back(left);
      neighbors_.push_back(right);
      //std::cout << " Initial " << item_ << std::endl;
    }

    void point::compute(std::vector<hpx::naming::id_type> const& point_components)
    {
     // hpx::util::spinlock::scoped_lock l(mtx_);
      if ( !active_ ) return;
      typedef std::vector<hpx::lcos::future< std::size_t > > lazy_results_type;
      lazy_results_type lazy_results;

      for (std::size_t i=0;i<neighbors_.size();i++) {
        lazy_results.push_back( stubs::point::get_item_async(point_components[neighbors_[i]]) );
      }

      std::vector<std::size_t> n;
      hpx::lcos::wait(lazy_results,n);  

      sum_ = item_ + n[0] + n[1];

      //std::cout << " Compute " << item_ << std::endl;
    }

    std::size_t point::get_item()
    {
      //std::cout << " Get_item " << item_ << std::endl;
      //hpx::util::spinlock::scoped_lock l(mtx_);
      if ( !active_ ) return 0;
      return item_;
    }

    void point::remove_item(std::size_t replace,std::size_t substitute)
    {
      //std::cout << " Remove_item " << item_ << std::endl;
      //hpx::util::spinlock::scoped_lock l(mtx_);
      if ( item_ == replace ) {
        active_ = false;
        neighbors_.resize(0);
      }
      for (std::size_t i=0;i<neighbors_.size();i++) {
        if (neighbors_[i] == replace) neighbors_[i] = substitute;
      }
    }
}}

