// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_AUX_ARBITRARY_DISTRIBUTION_20100917T1416)
#define PXGL_AUX_ARBITRARY_DISTRIBUTION_20100917T1416

#include <hpx/hpx.hpp>

#define LADIST_(lvl, str) YAP(lvl, " [ARB_DIST] " << str)

namespace pxgl { namespace xua {
  template <typename LocalityId, typename Region>
  class arbitrary_distribution
  {
  public:
    typedef LocalityId locality_id_type;
    typedef std::vector<LocalityId> locality_ids_type;
    typedef typename Region::index_type index_type;

    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    arbitrary_distribution()
      : region_(0),
        locales_(locality_ids_type()),
        num_locales_(0),
        map_(0),
        was_extended_(false),
        is_mapped_(false)
    {}

    arbitrary_distribution(locality_ids_type const & locales, 
                           Region const & region)
      : region_(region),
        locales_(locales),
        num_locales_(locales_.size()),
        map_(0),
        was_extended_(false),
        is_mapped_(false)
    {}

    arbitrary_distribution(locality_ids_type const & locales)
      : region_(0),
        locales_(locales),
        num_locales_(locales_.size()),
        map_(0),
        was_extended_(false),
        is_mapped_(false)
    {}

    locality_ids_type const & coverage(void) const
    {
      return locales_;
    }

    sizes_type const & map(void) const
    {
      return map_;
    }

    void extend(size_type locale_id)
    {
      was_extended_ = true;
      map_.push_back(locale_id);
    }

    void remap(sizes_type const & new_map)
    {
      was_extended_ = true;
      map_ = new_map;
    }

    size_type member_id(size_type index)
    {
      sizes_type::const_iterator mit =
          find(map_.begin(), map_.end(), locale_id(index));

      if (map_.end() == mit)
      {
        return size();
      }
      else
      {
        return mit - map_.begin();
      }
    }

    // Adjusts the view of the distribution for a dynamic component
    void finalize_coverage(void)
    {
      is_mapped_ = true;

      // Adjust localities
      {
        locality_ids_type new_locales;
        BOOST_FOREACH(size_type const & index, map_)
        {
          new_locales.push_back(locales_[index]);
        }
        locales_ = new_locales;
      }
    }

    Region const & region(void) const
    {
      return region_;
    }

    size_type locale_id(index_type const & i) const
    {
      if (!is_mapped_)
      {
        return i % num_locales_;
      }
      else
      {
        typename sizes_type::const_iterator index =
            find(map_.begin(), map_.end(), (i % num_locales_));
        if (index != map_.end())
        {
          return index - map_.begin();
        }
        else
        {
          assert(0);

          return 42;
        }
      }
    }

    size_type locale_id(locality_id_type const & there) const 
    {
      typename locality_ids_type::const_iterator index =
          find(locales_.begin(), locales_.end(), there);

      if (index == locales_.end())
      {
        return locales_.size();
      }
      else
      {
        return index - locales_.begin();
      }
    }

    locality_id_type const & locale(index_type const & i) const
    {
      if (!is_mapped_)
        return locales_[i % num_locales_];
      else
        return locales_[locale_id(i)];
    }

    size_type size(void) const
    {
      if (0 == size_)
        size_ = locales_.size();
        
      return size_;
    }

    template <typename Distribution>
    bool operator==(Distribution const & rhs) const
    {
      return locales_ == rhs.coverage();
    }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
      ar & region_ & locales_ & num_locales_ & map_ & was_extended_ 
          & is_mapped_;
    }

    // Data members
    Region region_;
    locality_ids_type locales_;

    size_type num_locales_;

    sizes_type map_;
    bool was_extended_;
    bool is_mapped_;

    mutable size_type size_;
  };
}}

#endif

