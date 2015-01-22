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

namespace hpx { namespace serialization {

  template <class T>
  void polymorphic_nonintrusive_factory::save(output_archive& ar, const T& t)
  {
    const std::string class_name = typeid(t).name();
    ar << class_name;

    map_.at(class_name).save_function(ar, &t);
  }

  template <class T>
  void polymorphic_nonintrusive_factory::load(input_archive& ar, T& t)
  {
    std::string class_name;
    ar >> class_name;

    map_.at(class_name).load_function(ar, &t);
  }

  template <class T>
  T* polymorphic_nonintrusive_factory::load(input_archive& ar)
  {
    std::string class_name;
    ar >> class_name;

    const function_bunch_type& bunch = map_.at(class_name);
    T* t = static_cast<T*>(bunch.create_function());

    bunch.load_function(ar, t);
    return t;
  }

}}

#endif
