//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_IMPL_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_NONINTRUSIVE_FACTORY_IMPL_HPP

#include <hpx/runtime/serialization/detail/polymorphic_nonintrusive_factory.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/runtime/serialization/string.hpp>

namespace hpx { namespace serialization { namespace detail
{
   template <class T>
   void polymorphic_nonintrusive_factory::save(output_archive& ar, const T& t)
   {
       // It's safe to call typeid here. The typeid(t) return value is
       // only used for local lookup to the portable string that goes over the
       // wire
       const std::string class_name = typeinfo_map_.at(typeid(t).name());
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
       T* t = static_cast<T*>(bunch.create_function(ar));

       return t;
   }

}}}

#endif
