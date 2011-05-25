// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_GRAPHS_SIGNAL_VALUE_20110217T0959)
#define PXGL_GRAPHS_SIGNAL_VALUE_20110217T0959

#include <hpx/hpx.hpp>
#include <hpx/hpx_fwd.hpp>

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/hpx.hpp"
#include "../../pxgl/util/component.hpp"

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server {
  //////////////////////////////////////////////////////////////////////////////
  struct signal_value
  {
    //typedef int size_type;
    typedef unsigned long size_type;

    signal_value()
    {}

    signal_value(size_type const & order, 
                 size_type const & size)
      : order_(order),
        size_(size)
    {}

    size_type order(void) const { return order_; }
    size_type size(void) const { return size_; }

  private:
    // Serialization support
    friend class boost::serialization::access;

    template <typename Archive>
      void serialize(Archive& ar, const unsigned int)
      {
        ar & order_ & size_;
      }

    // Data members
    size_type order_;
    size_type size_;
  };
  typedef signal_value signal_value_type;
}}}

#endif

