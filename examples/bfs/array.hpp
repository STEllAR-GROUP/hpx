//  Copyright (c) 2009-2011 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(ARRAY_AUG_22_2011_1042AM)
#define ARRAY_AUG_22_2011_1042AM

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <valarray>
#include <vector>

#include <hpx/exception.hpp>
#include <hpx/util/stringstream.hpp>

namespace bfsg {

template<typename T>

class array {
public:

  // creates an empty rows array
  array() 
    : istart_(0), jstart_(0), kstart_(0), 
      isize_(0), jsize_(0), ksize_(0), vsize_(0), data_()
  {}

  std::size_t size() const
  {
    return vsize_;
  }

  std::size_t isize() const
  {
    return isize_;
  }

  std::size_t jsize() const
  {
    return jsize_;
  }

  std::size_t ksize() const
  {
    return ksize_;
  }

  std::size_t data_size() const
  {
    return data_.size();
  }

  void resize(std::size_t s,std::size_t t,std::size_t u) 
  {
    vsize_ = s*t*u;
    isize_ = s;
    jsize_ = t;
    ksize_ = u;
    istart_ = 0;
    jstart_ = 0;
    kstart_ = 0;
    data_.resize(s*t*u); 
  }

  std::valarray<T> slicer(std::size_t face,std::size_t depth) const 
  {
    if ( face > 5 ) {
      hpx::util::osstream strm;
      strm << "face error, face(" << face << ")";
      HPX_THROW_EXCEPTION(hpx::bad_parameter
        , "array1d::slicer"
        , hpx::util::osstream_get_string(strm));
    }

    std::size_t start;
    std::size_t lengths[2];
    std::size_t strides[2];
    if ( face == 0 ) {
      // min i face
      start=0;
      lengths[0]= ksize_*jsize_;
      lengths[1]= depth;
      strides[0]= isize_;
      strides[1]= 1;
    } else if ( face == 1 ) {
      // max i face
      start=isize_-depth;
      lengths[0]= ksize_*jsize_;
      lengths[1]= depth;
      strides[0]= isize_;
      strides[1]= 1;
    } else if ( face == 2 ) {
      // min j face
      start=0;
      lengths[0]= ksize_;
      lengths[1]= depth*isize_;
      strides[0]= jsize_*isize_;
      strides[1]= 1;
    } else if ( face == 3 ) {
      // max j face
      start=jsize_*isize_-isize_*depth;
      lengths[0]= ksize_;
      lengths[1]= depth*isize_;
      strides[0]= jsize_*isize_;      
      strides[1]= 1;
    } else if ( face == 4 ) {
      // min k face
      start=0;
      lengths[0]= jsize_*depth;
      lengths[1]= isize_;
      strides[0]= isize_;      
      strides[1]= 1;
    } else if ( face == 5 ) {
      // max k face
      start=isize_*jsize_*ksize_ - depth*isize_*jsize_;
      lengths[0]= jsize_*depth;
      lengths[1]= isize_;
      strides[0]= isize_;      
      strides[1]= 1;
    }

    std::gslice mygslice (start,std::valarray<size_t>(lengths,2),std::valarray<size_t>(strides,2));
    std::valarray<T> result = data_[mygslice];

    return result;
  }

  array & operator=(array const& other)
  {
    if (this != &other) 
    {
      resize(other.isize(),other.jsize(),other.ksize());
      for (std::size_t i=0;i<other.size();i++) {
        data_[i] = other.data_[i];
      }
      istart_ = other.istart_;
      jstart_ = other.jstart_;
      kstart_ = other.kstart_;
      vsize_ = other.vsize_;
    }
    return *this;
  }

  // basic item reference
  T & operator[](std::size_t i) 
  { 
    if (i >= 0 && i < data_.size() ) {
        return data_[i]; 
    } else {
        hpx::util::osstream strm;
        strm << "array out of bounds, index(" << i << "), "
             << "datasize(" << data_.size() << ")";
        HPX_THROW_EXCEPTION(hpx::bad_parameter
          , "array1d::operator[]"
          , hpx::util::osstream_get_string(strm));
    }
  }

  // basic item reference
  T & operator()(std::size_t i,std::size_t j,std::size_t k) 
  { 
     if(i-istart_ >= 0 && i-istart_ < isize_ &&
        j-jstart_ >= 0 && j-jstart_ < jsize_ &&
        k-kstart_ >= 0 && k-kstart_ < ksize_ &&
        (i-istart_) + isize_*(j-jstart_ + jsize_*( k-kstart_) ) < data_.size() ) {
       return data_[(i-istart_) + isize_*(j-jstart_ + jsize_*( k-kstart_) )]; 
     } 

     hpx::util::osstream strm;
     strm << "array out of bounds, index(" << i << "), "
          << "datasize(" << data_.size() << ")\n"
          << "  istart " << istart_ << "\n"
          << "  j == " << j << "\n"
          << "  jstart == " << jstart_ << "\n"
          << "  k == " << k << "\n"
          << "  kstart == " << kstart_ << "\n"
          << "  isize == " << isize_ << "\n"
          << "  jsize == " << jsize_ << "\n"
          << "  (i-istart_) + isize_*(j-jstart_ + jsize_*( k-kstart_) ) == "
          << (i-istart_) + isize_*(j-jstart_ + jsize_*( k-kstart_) );
     HPX_THROW_EXCEPTION(hpx::bad_parameter
       , "array1d::operator()"
       , hpx::util::osstream_get_string(strm));
  }

  template<class Archive>
  void do_save(Archive & ar,std::size_t face,std::size_t depth) const
  {
    BOOST_ASSERT(vsize_ == data_.size());
    if ( face > 6 ) {
        hpx::util::osstream strm;
        strm << "face error, face(" << face << ")";
        HPX_THROW_EXCEPTION(hpx::bad_parameter
          , "array1d::do_save"
          , hpx::util::osstream_get_string(strm));
    }
    // NOTE: vsize was serialized uninitialized. I changed it to be assigned
    // the value of vsize_, pretty sure this is correct.
    std::size_t istart,jstart,kstart,isize,jsize,ksize,vsize = vsize_;
    if ( face == 0 ) {
      // min i face
      istart = 0;
      jstart = 0;
      kstart = 0;
      isize = depth;
      jsize = jsize_;
      ksize = ksize_;
    } else if ( face == 1 ) {
      // max i face
      istart = isize_-depth;
      jstart = 0;
      kstart = 0;
      isize = depth;
      jsize = jsize_;
      ksize = ksize_;
    } else if ( face == 2 ) {
      // min j face
      istart = 0;
      jstart = 0;
      kstart = 0;
      isize = isize_;
      jsize = depth;
      ksize = ksize_;
    } else if ( face == 3 ) {
      // max j face
      istart = 0;
      jstart = jsize_-depth;
      kstart = 0;
      isize = isize_;
      jsize = depth;
      ksize = ksize_;
    } else if ( face == 4 ) {
      // min k face
      istart = 0;
      jstart = 0;
      kstart = 0;
      isize = isize_;
      jsize = jsize_;
      ksize = depth;
    } else if ( face == 5 ) {
      // max k face
      istart = 0;
      jstart = 0;
      kstart = ksize_ - depth;
      isize = isize_;
      jsize = jsize_;
      ksize = depth;
    } else {
      // full 3-d mesh
      istart = 0;
      jstart = 0;
      kstart = 0;
      isize = isize_;
      jsize = jsize_;
      ksize = ksize_;
    }
    ar & istart;
    ar & jstart;
    ar & kstart;
    ar & isize;
    ar & jsize;
    ar & ksize;
    ar & vsize;
    //std::size_t s = data_.size();
    //ar & s; 
    if ( face < 6 ) {
      std::valarray<T> const& d = slicer(face,depth);
      ar & d;
    } else {
      ar & data_;
    }
  }

  template<class Archive>
  void do_load(Archive & ar) 
  {
    ar & istart_;
    ar & jstart_;
    ar & kstart_;
    ar & isize_;
    ar & jsize_;
    ar & ksize_;
    ar & vsize_;
    ar & data_;
  }

protected:

  std::size_t istart_;
  std::size_t jstart_;
  std::size_t kstart_;
  std::size_t isize_;
  std::size_t jsize_;
  std::size_t ksize_;
  std::size_t vsize_;
  std::valarray<T> data_;

};
}
#endif
