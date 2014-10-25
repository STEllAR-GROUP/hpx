//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_SERIALIZE_HPP
#define HPX_SERIALIZATION_SERIALIZE_HPP

#include <hpx/config.hpp>

#include <hpx/serialization/access.hpp>
#include <hpx/serialization/archive.hpp>

#include <boost/type_traits/add_const.hpp>
#include <boost/utility/enable_if.hpp>

#define HPX_SERIALIZATION_SPLIT_MEMBER()                                        \
    void serialize(hpx::serialization::input_archive & ar, unsigned)            \
    {                                                                           \
        load(ar, 0);                                                            \
    }                                                                           \
    void serialize(hpx::serialization::output_archive & ar, unsigned) const     \
    {                                                                           \
        save(ar, 0);                                                            \
    }                                                                           \
/**/
#define HPX_SERIALIZATION_SPLIT_FREE(T)                                         \
    void serialize(hpx::serialization::input_archive & ar, T & t, unsigned)     \
    {                                                                           \
        load(ar, t, 0);                                                         \
    }                                                                           \
    void serialize(hpx::serialization::output_archive & ar, T & t, unsigned)\
    {                                                                           \
        save(ar, const_cast<typename boost::add_const<T>::type &>(t)            \
            , 0);                                                               \
    }                                                                           \
/**/

namespace hpx { namespace serialization {
    namespace detail {
        template <typename Archive, typename T>
        void invoke(archive<Archive> & ar, T & t)
        {
            ar.invoke(t);
        }
    }

    template <typename Archive, typename T>
    void serialize(Archive & ar, T & t, unsigned)
    {
        access::serialize(ar, t, 0);
    }

    template <typename T>
    output_archive & operator<<(output_archive & ar, T const & t)
    {
        detail::invoke(ar, t);
        return ar;
    }

    template <typename T>
    input_archive & operator>>(input_archive & ar, T & t)
    {
        detail::invoke(ar, t);
        return ar;
    }

    template <typename Archive, typename T>
    Archive & operator&(Archive & ar, T & t)
    {
        ar.invoke(t);
        return ar;
    }
}}

#include <hpx/serialization/base_object.hpp>

#endif
