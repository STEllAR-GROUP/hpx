//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM)
#define HPX_TRAITS_SERIALIZE_AS_FUTURE_AUG_08_2014_0853PM

#include <hpx/traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>

#include <hpx/lcos/wait_all.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Future, typename Enable>
    struct serialize_as_future
      : boost::mpl::false_
    {
        static void call(Future& f) {}
    };

    template <typename T>
    struct serialize_as_future<T const>
      : serialize_as_future<T>
    {};

    template <typename T>
    struct serialize_as_future<T&>
      : serialize_as_future<T>
    {};

    template <typename T>
    struct serialize_as_future<T&&>
      : serialize_as_future<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Future>
    struct serialize_as_future<Future
        , typename boost::enable_if<is_future<Future> >::type>
      : boost::mpl::true_
    {
        static void call(Future& f)
        {
            hpx::lcos::wait_all(f);
        }
    };

    template <typename Range>
    struct serialize_as_future<Range
        , typename boost::enable_if<is_future_range<Range> >::type>
      : boost::mpl::true_
    {
        static void call(Range& r)
        {
            hpx::lcos::wait_all(r);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct serialize_as_future_helper
        {
            template <typename T>
            void operator()(T& t) const
            {
                serialize_as_future<T>::call(t);
            }
        };
    }

    template <>
    struct serialize_as_future<util::tuple<> >
      : boost::mpl::false_
    {
        static void call(util::tuple<>& t) {}
    };

#define HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE_ELEM(Z, N, D)                    \
     || serialize_as_future<BOOST_PP_CAT(T, N)>::value                        \
    /**/

#define HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE(Z, N, D)                         \
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>                            \
    struct serialize_as_future<util::tuple<BOOST_PP_ENUM_PARAMS(N, T)> >      \
      : boost::mpl::bool_<false                                               \
            BOOST_PP_REPEAT(N, HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE_ELEM, _)  \
        >                                                                     \
    {                                                                         \
        static void call(util::tuple<BOOST_PP_ENUM_PARAMS(N, T)>& t)          \
        {                                                                     \
            boost::fusion::for_each(t, detail::serialize_as_future_helper()); \
        }                                                                     \
    };                                                                        \
    /**/

    BOOST_PP_REPEAT_FROM_TO(
        1, BOOST_PP_INC(HPX_TUPLE_LIMIT),
        HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE, _
    )

#undef HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE
#undef HPX_TRAITS_SERIALIZE_AS_FUTURE_TUPLE_ELEM
}}

#endif

