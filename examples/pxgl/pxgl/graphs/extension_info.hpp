// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_GRAPHS_EXTENSION_INFO_20110217T1513)
#define PXGL_GRAPHS_EXTENSION_INFO_20110217T1513

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>

#include <pxgl/pxgl.hpp>
#include <pxgl/util/hpx.hpp>
#include <pxgl/util/component.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server {
  //////////////////////////////////////////////////////////////////////////////
  struct extension_info 
  {
    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    extension_info()
    {}

    extension_info(
        ids_type const & sibling_ids, 
        sizes_type const & coverage_map)
      : sibling_ids_(sibling_ids),
        coverage_map_(coverage_map)
    {}

    ids_type const & sibling_ids(void) const { return sibling_ids_; }
    sizes_type const & coverage_map(void) const { return coverage_map_; }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
      void serialize(Archive& ar, const unsigned int)
      {
        ar & sibling_ids_ & coverage_map_;
      }

    // Data members
    ids_type sibling_ids_;
    sizes_type coverage_map_;
  };
  typedef extension_info extension_info_type;
}}}

#endif

