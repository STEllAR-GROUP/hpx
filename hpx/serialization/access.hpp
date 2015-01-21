//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_ACCESS_HPP
#define HPX_SERIALIZATION_ACCESS_HPP

#include <hpx/config.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/traits/polymorphic_traits.hpp>

#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/eval_if.hpp>

namespace hpx { namespace serialization {

  namespace detail {

    template <class Archive, class T>
    BOOST_FORCEINLINE void serialize_force_adl(Archive& ar, T& t, unsigned)
    {
      serialize(ar, t, 0);
    }
  }

  struct access
  {
    template <class T>
    class serialize_dispatcher
    {
      struct intrusive_polymorphic
      {
        // both following template functions are viable
        // to call right overloaded function according to T constness
        // and to prevent calling templated version of serialize function
        static void call(hpx::serialization::input_archive& ar, T& t, unsigned)
        {
          t.serialize(ar, 0);
        }

        static void call(hpx::serialization::output_archive& ar, const T& t, unsigned)
        {
          t.serialize(ar, 0);
        }
      };

      struct non_intrusive_polymorphic
      {
        // this additional indirection level is needed to
        // force ADL on the second phase of template lookup.
        // call of serialize function directly from base_object
        // finds only serialize-member function and doesn't
        // perfrom ADL
        template <class Archive>
        static void call(Archive& ar, T& t, unsigned)
        {
          detail::serialize_force_adl(ar, t, 0);
        }
      };

      struct usual
      {
        template <class Archive>
        static void call(Archive& ar, T& t, unsigned)
        {
          t.serialize(ar, 0);
        }
      };

    public:

      typedef typename boost::mpl::eval_if<
        hpx::traits::is_intrusive_polymorphic<T>,
          boost::mpl::identity<intrusive_polymorphic>,
          boost::mpl::eval_if<
            hpx::traits::is_nonintrusive_polymorphic<T>,
              boost::mpl::identity<non_intrusive_polymorphic>,
              boost::mpl::identity<usual>
          >
      >::type type;
    };

    template <class Archive, class T>
    static void serialize(Archive& ar, T& t, unsigned)
    {
      serialize_dispatcher<T>::type::call(ar, t, 0);
    }

    template <typename Archive, typename T> BOOST_FORCEINLINE
    static typename boost::disable_if<boost::is_polymorphic<T> >::type
    save_base_object(Archive & ar, const T & t, unsigned)
    {
        t.serialize(ar, 0);
    }

    template <typename Archive, typename T> BOOST_FORCEINLINE
    static typename boost::enable_if<boost::is_polymorphic<T> >::type
    save_base_object(Archive & ar, const T & t, unsigned)
    {
        // explicitly specify virtual function
        // to avoid infinite recursion
        t.T::save(ar, 0);
    }

    template <typename Archive, typename T> BOOST_FORCEINLINE
    static typename boost::disable_if<boost::is_polymorphic<T> >::type
    load_base_object(Archive & ar, T & t, unsigned)
    {
        t.serialize(ar, 0);
    }

    template <typename Archive, typename T> BOOST_FORCEINLINE
    static typename boost::enable_if<boost::is_polymorphic<T> >::type
    load_base_object(Archive & ar, T & t, unsigned)
    {
        // explicitly specify virtual function
        // to avoid infinite recursion
        t.T::load(ar, 0);
    }

    template <typename T> BOOST_FORCEINLINE
    static boost::uint64_t get_hash(const T* t)
    {
      return t->hpx_serialization_get_hash();
    }
  };
}}

#endif
