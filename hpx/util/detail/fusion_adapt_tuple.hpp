//  Copyright (c) 2001-2011 Joel de Guzman
//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUSION_ADAPT_TUPLE_HPP
#define HPX_UTIL_DETAIL_FUSION_ADAPT_TUPLE_HPP

#include <hpx/config.hpp>
#include <hpx/util/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
/// tag_of.hpp
#include <boost/fusion/support/tag_of_fwd.hpp>

namespace hpx { namespace util { namespace detail
{
    struct tuple_tag;
}}}

namespace boost { namespace fusion
{
    struct fusion_sequence_tag;

    namespace traits
    {
        template <typename ...Ts>
        struct tag_of< ::hpx::util::tuple<Ts...>, void>
        {
            typedef ::hpx::util::detail::tuple_tag type;
        };
    }
}}

namespace boost { namespace mpl
{
    template <typename>
    struct sequence_tag;

    template <typename ...Ts>
    struct sequence_tag< ::hpx::util::tuple<Ts...> >
    {
        typedef fusion::fusion_sequence_tag type;
    };

    template <typename ...Ts>
    struct sequence_tag< ::hpx::util::tuple<Ts...> const>
    {
        typedef fusion::fusion_sequence_tag type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
/// is_sequence_impl.hpp
#include <boost/type_traits/integral_constant.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template <typename Tag>
        struct is_sequence_impl;

        template <>
        struct is_sequence_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
              : boost::true_type
            {};
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// is_view_impl.hpp
#include <boost/type_traits/integral_constant.hpp>

namespace boost { namespace fusion
{
    namespace extension
    {
        template <typename Tag>
        struct is_view_impl;

        template <>
        struct is_view_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename T>
            struct apply
              : boost::false_type
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
        template <typename T>
        struct category_of_impl;

        template <>
        struct category_of_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename T>
            struct apply
            {
                typedef random_access_traversal_tag type;
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// tuple_iterator.hpp
#include <boost/fusion/support/tag_of_fwd.hpp>
#include <boost/type_traits/integral_constant.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace util { namespace detail
{
    struct tuple_iterator_tag;

    template <typename Tuple, std::size_t Index>
    struct tuple_iterator
    {
        typedef Tuple tuple_type;
        static std::size_t const index = Index;
        typedef tuple_iterator<Tuple const, Index> identity;

        explicit tuple_iterator(Tuple& tuple)
          : tuple_(tuple)
        {}

        Tuple& tuple_;
    };

    template <typename Tuple, std::size_t Index>
    typename tuple_element<Index, Tuple>::type&
    operator*(tuple_iterator<Tuple, Index> i)
    {
        return util::get<Index>(i.tuple_);
    }
}}}

namespace boost { namespace fusion
{
    namespace traits
    {
        template <typename Tuple, std::size_t Index>
        struct tag_of< ::hpx::util::detail::tuple_iterator<Tuple, Index>, void>
        {
            typedef ::hpx::util::detail::tuple_iterator_tag type;
        };
    }

    struct random_access_traversal_tag;

    namespace extension
    {
        template <typename T>
        struct category_of_impl;

        template <>
        struct category_of_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename T>
            struct apply
            {
                typedef random_access_traversal_tag type;
            };
        };

        template <typename Tag>
        struct value_of_impl;

        template <>
        struct value_of_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename Iterator>
            struct apply
              : ::hpx::util::tuple_element<
                    Iterator::index
                  , typename Iterator::tuple_type
                >
            {};
        };

        template <typename Tag>
        struct deref_impl;

        template <>
        struct deref_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename Iterator>
            struct apply
            {
                static int const index = Iterator::index;
                typedef typename
                    ::hpx::util::tuple_element<
                        Iterator::index
                      , typename Iterator::tuple_type
                    >::type
                    element;
                typedef element& type;

                static type call(Iterator const& iter)
                {
                    return ::hpx::util::get<index>(iter.tuple_);
                }
            };
        };

        template <typename Tag>
        struct advance_impl;

        template <>
        struct advance_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename Iterator, typename I>
            struct apply
            {
                static int const index = Iterator::index;
                typedef typename Iterator::tuple_type tuple_type;
                typedef ::hpx::util::detail::tuple_iterator<
                    tuple_type, index + I::value> type;

                static type call(Iterator const& i)
                {
                    return type(i.tuple_);
                }
            };
        };

        template <typename Tag>
        struct next_impl;

        template <>
        struct next_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename Iterator>
            struct apply
              : advance_impl< ::hpx::util::detail::tuple_iterator_tag>::
                    template apply<Iterator, boost::integral_constant<int, 1> >
            {};
        };

        template <typename Tag>
        struct prior_impl;

        template <>
        struct prior_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename Iterator>
            struct apply
              : advance_impl< ::hpx::util::detail::tuple_iterator_tag>::
                    template apply<Iterator, boost::integral_constant<int, -1> >
            {};
        };

        template <typename Tag>
        struct equal_to_impl;

        template <>
        struct equal_to_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename I1, typename I2>
            struct apply
              : boost::integral_constant<
                    bool,
                    std::is_same<typename I1::identity, typename I2::identity>::value
                >
            {};
        };

        template <typename Tag>
        struct distance_impl;

        template <>
        struct distance_impl< ::hpx::util::detail::tuple_iterator_tag>
        {
            template <typename First, typename Last>
            struct apply
            {
                typedef boost::integral_constant<
                    int,
                    Last::index - First::index
                > type;

                static type call(First const&, Last const&)
                {
                    return type();
                }
            };
        };
    }
}}

///////////////////////////////////////////////////////////////////////////////
/// at_impl.hpp

namespace boost { namespace fusion
{
    namespace extension
    {
        template <typename T>
        struct at_impl;

        template <>
        struct at_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence, typename I>
            struct apply
            {
                typedef typename
                    ::hpx::util::tuple_element<I::value, Sequence>::type
                    element;
                typedef element& type;

                static type call(Sequence& seq)
                {
                    return ::hpx::util::get<I::value>(seq);
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
        template <typename T>
        struct begin_impl;

        template <>
        struct begin_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
            {
                typedef ::hpx::util::detail::tuple_iterator<Sequence, 0> type;

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
        template <typename T>
        struct end_impl;

        template <>
        struct end_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
            {
                static int const size = ::hpx::util::tuple_size<Sequence>::value;
                typedef ::hpx::util::detail::tuple_iterator<Sequence, size> type;

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
#include <boost/mpl/size_t.hpp>

#include <cstddef>

namespace boost { namespace fusion
{
    namespace extension
    {
        template <typename T>
        struct size_impl;

        template <>
        struct size_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence>
            struct apply
              : boost::integral_constant<
                    std::size_t,
                    ::hpx::util::tuple_size<Sequence>::value
                >
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
        template <typename T>
        struct value_at_impl;

        template <>
        struct value_at_impl< ::hpx::util::detail::tuple_tag>
        {
            template <typename Sequence, typename I>
            struct apply
              : ::hpx::util::tuple_element<I::value, Sequence>
            {};
        };
    }
}}

#endif
