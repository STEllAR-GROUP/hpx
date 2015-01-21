//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_POLYMORPHIC_TRAITS_HPP
#define HPX_TRAITS_POLYMORPHIC_TRAITS_HPP

#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace traits {

  namespace detail {

    struct intrusive_helper
    {
      boost::uint64_t hpx_serialization_get_hash() const;
    };

    template <class T>
    struct intrusive_helper_composed: T, intrusive_helper {};

    template <boost::uint64_t (intrusive_helper::*) () const>
    struct member_function_holder {};

    template <class T, class Ambiguous =
      detail::member_function_holder<&detail::intrusive_helper::hpx_serialization_get_hash> >
    struct is_intrusive_polymorphic_imp: boost::mpl::true_ {};

    template <class T>
    struct is_intrusive_polymorphic_imp<T, 
      detail::member_function_holder<&detail::intrusive_helper_composed<T>::hpx_serialization_get_hash> >
    : boost::mpl::false_ {};

  } // namespace detail

  template <class T, class Enable = void>
  struct is_intrusive_polymorphic: boost::mpl::false_ {};

  template <class T>
  struct is_intrusive_polymorphic<T,
    typename boost::enable_if<boost::is_class<T> >::type>:
      detail::is_intrusive_polymorphic_imp<T> {};

  template <class T>
  struct is_nonintrusive_polymorphic:
    boost::mpl::false_ {};

}}

#define HPX_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                            \
  namespace hpx { namespace traits {                                          \
    template <>                                                               \
    struct is_nonintrusive_polymorphic<Class>:                                \
      boost::mpl::true_ {};                                                   \
  }}                                                                          \
/**/

#endif
