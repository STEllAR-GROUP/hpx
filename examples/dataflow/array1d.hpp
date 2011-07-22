//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson

#if !defined(ARRAY1D_JUL_22_2011_1042AM)
#define ARRAY1D_JUL_22_2011_1042AM

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <valarray>
#include <vector>

template<typename T>

class array1d {
public:

  // creates an empty rows array1d
  array1d() : start_(0),vsize_(0) {}

  std::size_t size() const
  {
    return vsize_;
  }

  void resize(std::size_t s) 
  {
    vsize_ = s;
    start_ = 0;
    data_.resize(s); 
  }

  T & operator=(const T & other)
  {
    if (this != &other) 
    {
      this.resize(other.size());
      for (std::size_t i=0;i<other.size();i++) {
        this[i] = other[i];
      }
    }
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
    std::valarray<double> const& d = data_[std::slice(slice_start,slice_end-slice_start,1)];
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
