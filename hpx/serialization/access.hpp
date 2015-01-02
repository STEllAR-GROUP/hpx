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

#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace serialization {
    struct access
    {
        template <typename Archive, typename T> BOOST_FORCEINLINE
        static typename boost::disable_if<boost::is_polymorphic<T> >::type
        serialize(Archive & ar, T & t, unsigned)
        {
            t.serialize(ar, 0);
        }

        // both following template functions are viable
        // to call right overloaded function based on T constness
        template <typename T> BOOST_FORCEINLINE
        static typename boost::enable_if<boost::is_polymorphic<T> >::type
        serialize(hpx::serialization::input_archive & ar, T & t, unsigned)
        {
            t.serialize(ar, 0);
        }

        template <typename T> BOOST_FORCEINLINE
        static typename boost::enable_if<boost::is_polymorphic<T> >::type
        serialize(hpx::serialization::output_archive & ar, const T & t, unsigned)
        {
            t.serialize(ar, 0);
        }

        template <typename Archive, typename T> BOOST_FORCEINLINE
        static typename boost::disable_if<boost::is_polymorphic<T> >::type
        save_base_object(Archive & ar, const T & t, unsigned)
        {
            t.serialize(ar, 0);
        }

        template <typename Archive, typename T> BOOST_FORCEINLINE
        static typename boost::disable_if<boost::is_polymorphic<T> >::type
        load_base_object(Archive & ar, T & t, unsigned)
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
