// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_AUX_CONSTANT_DISTRIBUTION_20100915T1718)
#define PXGL_AUX_CONSTANT_DISTRIBUTION_20100915T1718

#include <hpx/hpx.hpp>

#define LCDIST_(lvl, str) YAP(lvl, " [CONST_DIST] " << str)

namespace pxgl { namespace xua {
  template <typename LocalityId, typename Region>
  class constant_distribution
  {
  public:
    typedef LocalityId locality_id_type;
    typedef std::vector<LocalityId> locality_ids_type;
    typedef typename Region::index_type index_type;

    typedef int size_type;

    constant_distribution()
      : region_(0),
        locales_(locality_ids_type())
    {}

    constant_distribution(
        locality_ids_type const & locales, 
        Region const & region)
      : region_(region),
        locales_(locales)
    {}

    constant_distribution(
        locality_id_type const & locale, 
        Region const & region)
      : region_(region),
        locales_(locality_ids_type(1))
    {
      locales_[0] = locale;
    }

    locality_ids_type const & coverage(void) const
    {
      return locales_;
    }

    locality_id_type locale(index_type i) const
    {
      if (region_.has_index(i))
        return locales_[0];
      else
        return locality_id_type();
    }

    Region const & region(void) const
    {
      return region_;
    }

    size_type size(void) const
    {
      return 1;
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
    Region region_;
    locality_ids_type locales_;
  };
}}

#endif

