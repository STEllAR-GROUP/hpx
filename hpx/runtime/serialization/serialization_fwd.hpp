//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_FWD_HPP
#define HPX_SERIALIZATION_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#include <type_traits>

namespace hpx { namespace serialization
{
    class access;
    struct input_archive;
    struct output_archive;
    struct binary_filter;

    template <typename T>
    output_archive & operator<<(output_archive & ar, T const & t);

    template <typename T>
    input_archive & operator>>(input_archive & ar, T & t);

    template <typename T>
    output_archive & operator&(output_archive & ar, T const & t);

    template <typename T>
    input_archive & operator&(input_archive & ar, T & t);
}}

#define HPX_SERIALIZATION_SPLIT_MEMBER()                                       \
    void serialize(hpx::serialization::input_archive & ar, unsigned)           \
    {                                                                          \
        load(ar, 0);                                                           \
    }                                                                          \
    void serialize(hpx::serialization::output_archive & ar, unsigned) const    \
    {                                                                          \
        save(ar, 0);                                                           \
    }                                                                          \
/**/

#define HPX_SERIALIZATION_SPLIT_FREE(T)                                        \
    HPX_FORCEINLINE                                                            \
    void serialize(hpx::serialization::input_archive & ar, T & t, unsigned)    \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    HPX_FORCEINLINE                                                            \
    void serialize(hpx::serialization::output_archive & ar, T & t, unsigned)   \
    {                                                                          \
        save(ar, const_cast<std::add_const<T>::type &>(t), 0);                 \
    }                                                                          \
/**/

#define HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(TEMPLATE, ARGS)                  \
    HPX_UTIL_STRIP(TEMPLATE)                                                   \
    HPX_FORCEINLINE                                                            \
    void serialize(hpx::serialization::input_archive & ar,                     \
            HPX_UTIL_STRIP(ARGS) & t, unsigned)                                \
    {                                                                          \
        load(ar, t, 0);                                                        \
    }                                                                          \
    HPX_UTIL_STRIP(TEMPLATE)                                                   \
    HPX_FORCEINLINE                                                            \
    void serialize(hpx::serialization::output_archive & ar,                    \
            HPX_UTIL_STRIP(ARGS) & t, unsigned)                                \
    {                                                                          \
        save(ar, const_cast<typename std::add_const                            \
                <HPX_UTIL_STRIP(ARGS)>::type &>(t), 0);                        \
    }                                                                          \
/**/

#endif // HPX_SERIALIZATION_FWD_HPP
