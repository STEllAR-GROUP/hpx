//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_IPP
#define HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_IPP

#include <hpx/serialization/polymorphic_nonintrusive_factory.hpp>

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <boost/fusion/include/at_c.hpp>

namespace hpx { namespace serialization {

  template <class T>
  void polymorphic_nonintrusive_factory::save(output_archive& ar, const T& t)
  {
    boost::uint64_t hash =
      hpx::util::jenkins_hash()(typeid(t).name());
    ar << hash;
    boost::fusion::at_c<0>(map_.at(hash))(ar, &t);
  }

  template <class T>
  void polymorphic_nonintrusive_factory::load(input_archive& ar, T& t)
  {
    boost::uint64_t hash = 0u;
    ar >> hash;

    HPX_ASSERT(hash);

    boost::fusion::at_c<1>(map_.at(hash))(ar, &t);
  }

  template <class T>
  T* polymorphic_nonintrusive_factory::load(input_archive& ar)
  {
    boost::uint64_t hash = 0u;
    ar >> hash;

    HPX_ASSERT(hash);

    const function_bunch_type& bunch = map_.at(hash);
    T* t = static_cast<T*>(boost::fusion::at_c<2>(bunch)());

    boost::fusion::at_c<1>(bunch)(ar, t);
    return t;
  }

}}

#endif
