//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_FWD_HPP
#define HPX_SERIALIZATION_FWD_HPP

namespace hpx { namespace serialization {

  struct input_archive;
  struct output_archive;

}}

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
    void serialize(hpx::serialization::output_archive & ar, T & t, unsigned)    \
    {                                                                           \
        save(ar, const_cast<typename boost::add_const<T>::type &>(t)            \
            , 0);                                                               \
    }                                                                           \
/**/

#endif // HPX_SERIALIZATION_FWD_HPP
