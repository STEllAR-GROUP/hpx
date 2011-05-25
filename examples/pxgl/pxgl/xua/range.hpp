// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_AUX_RANGE_20100915T1651)
#define PXGL_AUX_RANGE_20100915T1651

#define LRNG_(lvl, str) YAP(lvl, " [RANGE] " << str)

namespace pxgl { namespace xua {
  class range
  {
  public:
    typedef int size_type;
    typedef size_type index_type;

    range()
      : size_(0)
    {}

    range(size_type const & size)
      : size_(size)
    {};

    bool has_index(index_type const & i) const
    {
      return (0 <= i && i < size_);
    }

    size_type size(void) const
    {
      return size_;
    }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
      ar & size_;
    }

    // Data members
    size_type size_;
  };
}}

#endif

