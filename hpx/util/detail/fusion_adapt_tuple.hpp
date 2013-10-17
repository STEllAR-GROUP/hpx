//  Copyright (c) 2001-2011 Joel de Guzman
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUSION_ADAPT_TUPLE_HPP
#define HPX_UTIL_DETAIL_FUSION_ADAPT_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/preprocessor/min.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#   define HPX_FUSION_TUPLE_MAX BOOST_PP_MIN(BOOST_PP_ADD(HPX_PP_ROUND_UP(HPX_TUPLE_LIMIT), 3), HPX_MAX_LIMIT)
#else
#   define HPX_FUSION_TUPLE_MAX HPX_TUPLE_LIMIT
#endif

///////////////////////////////////////////////////////////////////////////////
/// tag_of.hpp
#include <boost/fusion/support/tag_of_fwd.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace hpx { namespace util { namespace detail
{
    struct tuple_tag;
}}}

namespace boost { namespace fusion
{
    struct fusion_sequence_tag;

    namespace traits
    {
        template <BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, typename T)>
        struct tag_of<
            hpx::util::tuple<BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, T)>
        >
        {
            typedef hpx::util::detail::tuple_tag type;
        };
    }
}}

namespace boost { namespace mpl
{
    template <typename>
    struct sequence_tag;

    template <BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, typename T)>
    struct sequence_tag<hpx::util::tuple<
        BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, T)>
    >
    {
        typedef fusion::fusion_sequence_tag type;
    };

    template <BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, typename T)>
    struct sequence_tag<hpx::util::tuple<
        BOOST_PP_ENUM_PARAMS(HPX_FUSION_TUPLE_MAX, T)> const
    >
    {
        typedef fusion::fusion_sequence_tag type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// is_sequence_impl.hpp
#include <boost/mpl/bool.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename Tag>
        struct is_sequence_impl;

        template<>
        struct is_sequence_impl<hpx::util::detail::tuple_tag>
        {
            template<typename Sequence>
            struct apply
              : mpl::true_
            {};
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// is_view_impl.hpp
#include <boost/mpl/bool.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename Tag>
        struct is_view_impl;

        template<>
        struct is_view_impl<hpx::util::detail::tuple_tag>
        {
            template<typename T>
            struct apply
              : mpl::false_
            {};
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// category_of_impl.hpp

namespace boost { namespace fusion
{
    struct random_access_traversal_tag;

    namespace extension
    {
        template<typename T>
        struct category_of_impl;

        template<>
        struct category_of_impl<hpx::util::detail::tuple_tag>
        {
            template<typename T>
            struct apply
            {
                typedef random_access_traversal_tag type;
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// hpx_tuple_iterator.hpp
#include <hpx/util/add_lvalue_reference.hpp>
#include <boost/fusion/iterator/iterator_facade.hpp>
#include <boost/mpl/int.hpp>

namespace boost { namespace fusion
{
    struct random_access_traversal_tag;

    template <typename Tuple, int Index>
    struct hpx_tuple_iterator_identity;

    template <typename Tuple, int Index>
    struct hpx_tuple_iterator
      : iterator_facade<
            hpx_tuple_iterator<Tuple, Index>
          , random_access_traversal_tag
        >
    {
        typedef Tuple tuple_type;
        static int const index = Index;
        typedef hpx_tuple_iterator_identity<
            typename add_const<Tuple>::type, Index>
            identity;

        explicit hpx_tuple_iterator(Tuple& tuple)
          : tuple(tuple) {}

        Tuple& tuple;

        template <typename Iterator>
        struct value_of
          : hpx::util::tuple_element<
                Iterator::index
              , typename Iterator::tuple_type
            >
        {};

        template <typename Iterator>
        struct deref
        {
            typedef typename 
                hpx::util::tuple_element<
                    Iterator::index
                  , typename Iterator::tuple_type
                >::type
                element;
            typedef typename
                hpx::util::add_lvalue_reference<element>::type
                type;

            static type call(Iterator const& iter)
            {
                return hpx::util::get<Index>(iter.tuple);
            }
        };

        template <typename Iterator, typename I>
        struct advance
        {
            static int const index = Iterator::index;
            typedef typename Iterator::tuple_type tuple_type;
            typedef hpx_tuple_iterator<tuple_type, index+I::value> type;

            static type call(Iterator const& i)
            {
                return type(i.tuple);
            }
        };

        template <typename Iterator>
        struct next
          : advance<Iterator, mpl::int_<1> >
        {};

        template <typename Iterator>
        struct prior
          : advance<Iterator, mpl::int_<-1> >
        {};

        template <typename I1, typename I2>
        struct equal_to
          : is_same<typename I1::identity, typename I2::identity>
        {};

        template <typename First, typename Last>
        struct distance
        {
            typedef mpl::int_<Last::index-First::index> type;

            static type call(First const&, Last const&)
            {
                return type();
            }
        };
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// at_impl.hpp
#include <hpx/util/add_lvalue_reference.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename T>
        struct at_impl;

        template <>
        struct at_impl<hpx::util::detail::tuple_tag>
        {
            template <typename Sequence, typename I>
            struct apply
            {
                typedef typename 
                    hpx::util::tuple_element<I::value, Sequence>::type
                    element;
                typedef typename
                    hpx::util::add_lvalue_reference<element>::type
                    type;

                static type call(Sequence& seq)
                {
                    return hpx::util::get<I::value>(seq);
                }
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// begin_impl.hpp

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename T>
        struct begin_impl;

        template <>
        struct begin_impl<hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
            {
                typedef hpx_tuple_iterator<Sequence, 0> type;

                static type call(Sequence& v)
                {
                    return type(v);
                }
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// end_impl.hpp

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename T>
        struct end_impl;

        template <>
        struct end_impl<hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
            {
                static int const size = hpx::util::tuple_size<Sequence>::value;
                typedef hpx_tuple_iterator<Sequence, size> type;

                static type call(Sequence& v)
                {
                    return type(v);
                }
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// size_impl.hpp
#include <boost/mpl/int.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename T>
        struct size_impl;

        template <>
        struct size_impl<hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
              : mpl::int_<hpx::util::tuple_size<Sequence>::value>
            {};
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// value_at_impl.hpp
namespace boost { namespace fusion
{
    namespace extension
    {
        template<typename T>
        struct value_at_impl;

        template <>
        struct value_at_impl<hpx::util::detail::tuple_tag>
        {
            template <typename Sequence, typename I>
            struct apply
              : hpx::util::tuple_element<I::value, Sequence>
            {};
        };
    }
}}

#undef HPX_FUSION_TUPLE_MAX

#endif
