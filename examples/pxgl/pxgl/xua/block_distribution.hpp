// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_AUX_BLOCK_DISTRIBUTION_20100917T1416)
#define PXGL_AUX_BLOCK_DISTRIBUTION_20100917T1416

#include <hpx/hpx.hpp>

#define LADIST_(lvl, str) YAP(lvl, " [ARB_DIST] " << str)

namespace pxgl { namespace xua {
  template <typename LocalityId, typename Region>
  class block_distribution
  {
  public:
    typedef LocalityId locality_id_type;
    typedef std::vector<LocalityId> locality_ids_type;
    typedef typename Region::index_type index_type;

    typedef int size_type;

    block_distribution()
      : region_(0),
        locales_(locality_ids_type())
    {}

    block_distribution(locality_ids_type locales, Region region)
      : region_(region),
        locales_(locales)
    {
      p_ = num_items / num_locales;
      q_ = num_items % num_locales;
      mid_ = (p_ + 1) * q_;

      LADIST_(info, "(" << p_ << ", " << q_ << ", " << mid_ << ")")
    }

    block_distribution(locality_ids_type locales)
      : region_(0),
        locales_(locales),
        num_locales_(locales_.size())
    {
      assert(0);
    }

    locality_ids_type const& coverage(void) const
    {
      return locales_;
    }

    Region region(void)
    {
      return region_;
    }

    size_type locale_id(index_type i)
    {
      return i % num_locales_;
    }

    locality_id_type locale(index_type i)
    {
      if (q_ == 0)
        return locales_[i / p_];
      else if (i < mid_)
        return locales_[i / (p_ + 1)];
      else
        return locales_[((i - mid_) / p_) + q_];
    }

    size_type size(void)
    {
      return locales_.size();
    }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
      ar & region_ & locales_;
    }

    // Data members
    locality_ids_type locales_;
    Region region_;

    size_type num_locales_;

    size_type p_, q_, mid_;
  };
}}

#endif

