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
#include <boost/preprocessor/iteration.hpp>
#include <boost/preprocessor/enum_params.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util { namespace detail { namespace adl_barrier
{
    template <typename T>
    struct has_serialize_mem_fun
    {
        typedef void (T::*SerializationFun)(int&, unsigned);

        template <SerializationFun>
        struct A {};

        template <typename Q>
        static boost::type_traits::yes_type
            has_serialize_mem_fun_tester(A<&Q::serialize>*);

        template <typename Q>
        static boost::type_traits::no_type has_serialize_mem_fun_tester(...);

        BOOST_STATIC_CONSTANT(bool, value =
            (sizeof(has_serialize_mem_fun_tester<T>(0)) ==
                sizeof(boost::type_traits::yes_type)));
    };

    template <typename T, typename IsFundamental = void>
    struct is_serializable_impl
    {
        BOOST_STATIC_CONSTANT(bool, value = has_serialize_mem_fun<T>::value);
    };

    template <typename T>
    struct is_serializable_impl<T,
        typename boost::enable_if<boost::is_fundamental<T> >::type>
    {
        BOOST_STATIC_CONSTANT(bool, value = true);
    };

/*
    ///////////////////////////////////////////////////////////////////////////
    struct tag {};

#define BOOST_PP_LOCAL_LIMITS (1, 10)
#define BOOST_PP_LOCAL_MACRO(N)                                               \
        template <                                                            \
            typename Archive,                                                 \
            template <BOOST_PP_ENUM_PARAMS(N, class I)> class T,              \
            BOOST_PP_ENUM_PARAMS(N, typename T)>                              \
        tag serialize(Archive& ar, T<BOOST_PP_ENUM_PARAMS(N, T)>&,            \
            const unsigned);                                                  \

#include BOOST_PP_LOCAL_ITERATE()

    template <typename T>
    struct is_non_intrusively_serializable_impl
    {
        tag operator,(tag, int);

        boost::type_traits::no_type check(tag);

        template <typename T>
        boost::type_traits::yes_type check(T const&);

        static typename boost::remove_cv<T>::type& x;
        static const int& archive;
        static const unsigned version;

        BOOST_STATIC_CONSTANT(bool, value =
            sizeof(check((serialize(archive, x, version), 0))) ==
                sizeof(boost::type_traits::yes_type));
    };
*/
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    BOOST_TT_AUX_BOOL_TRAIT_DEF1(is_intrusively_serializable, T,
        hpx::util::detail::adl_barrier::is_serializable_impl<T>::value)

/*
    BOOST_TT_AUX_BOOL_TRAIT_DEF1(is_non_intrusively_serializable, T,
        hpx::util::detail::adl_barrier::is_non_intrusively_serializable_impl<T>::value)

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_serializable
      : boost::mpl::or_<
            is_intrusively_serializable<T>
          , is_non_intrusively_serializable<T> >
    {};
*/
}}

#endif
