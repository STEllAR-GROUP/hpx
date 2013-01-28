//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Parts of this code were posted by Roman Perepelitsa and Chris Fairles here:
//  http://boost.2283326.n4.nabble.com/Serialization-is-serializable-trait-td2566933.html
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_IS_SERIALIZABLE_JAN_25_2013_0807PM)
#define HPX_UTIL_IS_SERIALIZABLE_JAN_25_2013_0807PM

#include <boost/serialization/serialization.hpp>
#include <boost/type_traits/remove_cv.hpp>
#include <boost/type_traits/detail/yes_no_type.hpp>
#include <boost/type_traits/detail/bool_trait_def.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util { namespace detail { namespace adl_barrier
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    struct has_serialize_mem_fun_getter
    {
        template <
            typename T
          , void (T::*)(Archive&, unsigned int const) =
                (void(T::*)(Archive&, unsigned int const))
                    &T::serialize<Archive> >
        struct get
        {};
    };

    template <typename T, typename Archive,
        typename Getter = has_serialize_mem_fun_getter<Archive> >
    struct has_serialize_mem_fun
    {
        template <typename Q>
        static boost::type_traits::yes_type
            has_serialize_mem_fun_tester(typename Getter::template get<Q>*);

        template <typename Q>
        static boost::type_traits::no_type
            has_serialize_mem_fun_tester(...);

        BOOST_STATIC_CONSTANT(bool, value =
            (sizeof(has_serialize_mem_fun_tester<T>(0)) ==
                sizeof(boost::type_traits::yes_type)));
    };

    template <typename T, typename Archive, typename IsFundamental = void>
    struct is_serializable_impl
    {
        typedef has_serialize_mem_fun<T, Archive> type;
        BOOST_STATIC_CONSTANT(bool, value = type::value);
    };

    template <typename T, typename Archive>
    struct is_serializable_impl<T, Archive,
        typename boost::enable_if<boost::is_fundamental<T> >::type>
    {
        BOOST_STATIC_CONSTANT(bool, value = true);
    };

    ///////////////////////////////////////////////////////////////////////////
    struct serialization_tag {};
    serialization_tag operator,(serialization_tag, int);

    template <typename Archive, typename T>
    serialization_tag serialize(Archive& ar, T&, const unsigned);

    template <typename T, typename Archive>
    struct is_non_intrusively_serializable_impl
    {
        static boost::type_traits::no_type check(serialization_tag);

        template <typename T>
        static boost::type_traits::yes_type check(T const&);

        static typename boost::remove_cv<T>::type& x;
        static Archive& archive;
        static const unsigned version;

        BOOST_STATIC_CONSTANT(bool, value =
            sizeof(check((serialize(archive, x, version), 0))) ==
                sizeof(boost::type_traits::yes_type));
    };
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    BOOST_TT_AUX_BOOL_TRAIT_DEF2(is_intrusively_serializable, T, Archive,
        (hpx::util::detail::adl_barrier::is_serializable_impl<T, Archive>::value))

    BOOST_TT_AUX_BOOL_TRAIT_DEF2(is_non_intrusively_serializable, T, Archive,
        (hpx::util::detail::adl_barrier::is_non_intrusively_serializable_impl<
            T, Archive>::value))

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Archive>
    struct is_serializable
      : boost::mpl::or_<
            is_intrusively_serializable<T, Archive>,
            is_non_intrusively_serializable<T, Archive> >
    {};
}}

#endif
