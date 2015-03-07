//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_FWD_HPP
#define HPX_SERIALIZATION_FWD_HPP

#include <hpx/config.hpp>

namespace hpx { namespace serialization {

    namespace detail
    {
        struct ptr_helper;
    }

    struct input_archive;
    struct output_archive;

    BOOST_FORCEINLINE
    void register_pointer(input_archive & ar, std::size_t pos, HPX_STD_UNIQUE_PTR<detail::ptr_helper> helper);

    template <typename Archive, typename T>
    void serialize(Archive & ar, T & t, unsigned);

    template <typename T>
    output_archive & operator<<(output_archive & ar, T const & t);

    template <typename T>
    input_archive & operator>>(input_archive & ar, T & t);

    template <typename T>
    output_archive & operator&(output_archive & ar, T const & t);

    template <typename T>
    input_archive & operator&(input_archive & ar, T & t);

}}

#define HPX_SERIALIZATION_SPLIT_MEMBER()                                            \
    void serialize(hpx::serialization::input_archive & ar, unsigned)                \
    {                                                                               \
        load(ar, 0);                                                                \
    }                                                                               \
    void serialize(hpx::serialization::output_archive & ar, unsigned) const         \
    {                                                                               \
        save(ar, 0);                                                                \
    }                                                                               \
/**/
#define HPX_SERIALIZATION_SPLIT_FREE(T)                                             \
    static void serialize(hpx::serialization::input_archive & ar, T & t, unsigned)  \
    {                                                                               \
        load(ar, t, 0);                                                             \
    }                                                                               \
    static void serialize(hpx::serialization::output_archive & ar, T & t, unsigned) \
    {                                                                               \
        save(ar, const_cast<typename boost::add_const<T>::type &>(t), 0);           \
    }                                                                               \
/**/

#endif // HPX_SERIALIZATION_FWD_HPP                                                  
