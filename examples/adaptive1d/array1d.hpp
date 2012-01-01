//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(ARRAY1D_AUG_22_2011_1042AM)
#define ARRAY1D_AUG_22_2011_1042AM

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <valarray>
#include <vector>
#include "stencil/stencil_data.hpp"

template<typename T>

class array1d {
public:

  // creates an empty rows array1d
  array1d() : start_(0),vsize_(0) {}

  std::size_t size() const
  {
    return vsize_;
  }

  std::size_t data_size() const
  {
    return data_.size();
  }

  void resize(std::size_t s)
  {
    vsize_ = s;
    start_ = 0;
    data_.resize(s);
  }

  array1d & operator=(array1d const& other)
  {
    if (this != &other)
    {
      resize(other.size());
      for (std::size_t i=0;i<other.size();i++) {
        data_[i] = other.data_[i];
      }
      start_ = other.start_;
      vsize_ = other.vsize_;
    }
    return *this;
  }

  // basic item reference
  T & operator[](std::size_t r)
  {
     BOOST_ASSERT(r-start_ >= 0 && r-start_ < data_.size() );
     return data_[r - start_];
  }
  // basic item retrieval
  T operator[](std::size_t r) const
  {
    BOOST_ASSERT(r-start_ >= 0 && r-start_ < data_.size() );
    return data_[r-start_];
  }

  template<class Archive>
  void do_save(Archive & ar,std::size_t slice_start,std::size_t slice_end) const
  {
    BOOST_ASSERT(vsize_ == data_.size());
    ar & slice_start;
    std::size_t s = data_.size();
    ar & s; // vsize
    std::valarray<T> const& d = data_[std::slice(slice_start,slice_end-slice_start,1)];
    ar & d;
  }

  template<class Archive>
  void do_load(Archive & ar)
  {
    ar & start_;
    ar & vsize_;
    ar & data_;
  }

protected:

  std::size_t start_;
  std::size_t vsize_;
  std::valarray<T> data_;

};
#endif
