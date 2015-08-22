//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_FWD_HPP
#define HPX_SERIALIZATION_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/pp_strip_parens.hpp>

#if defined(HPX_INTEL_VERSION) && ((__GNUC__ == 4 && __GNUC_MINOR__ == 4) \
           || HPX_INTEL_VERSION < 1400)
#include <boost/shared_ptr.hpp>
#else
#include <memory>
#endif

#include <boost/cstdint.hpp>

namespace hpx { namespace serialization
{
    namespace detail
    {
        struct ptr_helper;
#if defined(HPX_INTEL_VERSION) && ((__GNUC__ == 4 && __GNUC_MINOR__ == 4) \
           || HPX_INTEL_VERSION < 1400)
        typedef boost::shared_ptr<ptr_helper> ptr_helper_ptr;
#else
        typedef std::unique_ptr<ptr_helper> ptr_helper_ptr;
#endif
    }

    struct input_archive;
    struct output_archive;

    BOOST_FORCEINLINE
    void register_pointer(input_archive & ar, boost::uint64_t pos,
        detail::ptr_helper_ptr helper);

    template <typename Helper>
    Helper & tracked_pointer(input_archive & ar, boost::uint64_t pos);

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
    BOOST_FORCEINLINE                                                               \
    void serialize(hpx::serialization::input_archive & ar, T & t, unsigned)         \
    {                                                                               \
        load(ar, t, 0);                                                             \
    }                                                                               \
    BOOST_FORCEINLINE                                                               \
    void serialize(hpx::serialization::output_archive & ar, T & t, unsigned)        \
    {                                                                               \
        save(ar, const_cast<boost::add_const<T>::type &>(t), 0);                    \
    }                                                                               \
/**/
#define HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(TEMPLATE, ARGS)                       \
    HPX_UTIL_STRIP(TEMPLATE)                                                        \
    BOOST_FORCEINLINE                                                               \
    void serialize(hpx::serialization::input_archive & ar,                          \
            HPX_UTIL_STRIP(ARGS) & t, unsigned)                                     \
    {                                                                               \
        load(ar, t, 0);                                                             \
    }                                                                               \
    HPX_UTIL_STRIP(TEMPLATE)                                                        \
    BOOST_FORCEINLINE                                                               \
    void serialize(hpx::serialization::output_archive & ar,                         \
            HPX_UTIL_STRIP(ARGS) & t, unsigned)                                     \
    {                                                                               \
        save(ar, const_cast<typename boost::add_const                               \
                <HPX_UTIL_STRIP(ARGS)>::type &>(t), 0);                             \
    }                                                                               \
/**/

#endif // HPX_SERIALIZATION_FWD_HPP
