//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_INTRUSIVE_PTR_FEB_20_2012_1137AM)
#define HPX_UTIL_SERIALIZE_INTRUSIVE_PTR_FEB_20_2012_1137AM

#include <boost/config.hpp>
#include <boost/intrusive_ptr.hpp>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

// define macro to let users of these compilers do this
#define BOOST_SERIALIZATION_INTRUSIVE_PTR(T)                                  \
    BOOST_CLASS_VERSION(::boost::intrusive_ptr<T>, 1)                         \
    BOOST_CLASS_TRACKING(::boost::intrusive_ptr<T>,                           \
        ::boost::serialization::track_never)                                  \
    /**/

namespace boost { namespace serialization
{
    template <typename Archive, typename T>
    inline void save(Archive& ar, boost::intrusive_ptr<T> const& t,
        unsigned int const)
    {
        // The most common cause of trapping here would be serializing
        // something like intrusive_ptr<int>.  This occurs because int
        // is never tracked by default.  Wrap int in a trackable type
        BOOST_STATIC_ASSERT((tracking_level<T>::value != track_never));
        T const* ptr = t.get();
        ar << ptr;
    }

    template <typename Archive, typename T>
    inline void load(Archive& ar, boost::intrusive_ptr<T>& t,
        const unsigned int)
    {
        // The most common cause of trapping here would be serializing
        // something like shared_ptr<int>.  This occurs because int
        // is never tracked by default.  Wrap int in a trackable type
        BOOST_STATIC_ASSERT((tracking_level<T>::value != track_never));
        T* ptr;
        ar >> ptr;
        t.reset(ptr);
    }

    template <typename Archive, typename T>
    inline void serialize(Archive& ar, boost::intrusive_ptr<T>& t,
        unsigned int const version)
    {
        // correct shared_ptr serialization depends upon object tracking
        // being used.
        BOOST_STATIC_ASSERT(tracking_level<T>::value != track_never);
        boost::serialization::split_free(ar, t, version);
    }
}}

#endif // HPX_UTIL_SERIALIZE_INTRUSIVE_PTR_FEB_20_2012_1137AM
