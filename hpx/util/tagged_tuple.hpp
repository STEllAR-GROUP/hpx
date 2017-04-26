//  Copyright Eric Niebler 2013-2015
//  Copyright 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#if !defined(HPX_UTIL_TAGGED_TUPLE_DEC_23_2015_0123PM)
#define HPX_UTIL_TAGGED_TUPLE_DEC_23_2015_0123PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/identity.hpp>
#include <hpx/util/tagged.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct tagged_tuple
      : tagged<
            tuple<typename detail::tag_elem<Ts>::type...>,
            typename detail::tag_spec<Ts>::type...>
    {
        typedef tagged<
                tuple<typename detail::tag_elem<Ts>::type...>,
                typename detail::tag_spec<Ts>::type...
            > base_type;

        template <typename ...Ts_>
        tagged_tuple(Ts_ &&... ts)
          : base_type(std::forward<Ts_>(ts)...)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Tag, typename T>
        struct tagged_type
        {
            typedef typename decay<T>::type decayed_type;
            typedef typename hpx::util::identity<Tag(decayed_type)>::type type;
        };
    }

#if defined(HPX_HAVE_CXX11_EXPLICIT_VARIADIC_TEMPLATES)
    template <typename ...Tags, typename ...Ts>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
    make_tagged_tuple(Ts && ...ts)
    {
        typedef tagged_tuple<
                typename detail::tagged_type<Tags, Ts>::type...
            > result_type;

        return result_type(std::forward<Ts>(ts)...);
    }

    template <typename ...Tags, typename ...Ts>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_tuple<typename detail::tagged_type<Tags, Ts>::type...>
    make_tagged_tuple(tuple<Ts...> && t)
    {
        static_assert(sizeof...(Tags) == tuple_size<tuple<Ts...> >::value,
            "the number of tags must be identical to the size of the given tuple");

        typedef tagged_tuple<
                typename detail::tagged_type<Tags, Ts>::type...
            > result_type;

        return result_type(std::move(t));
    }
#else
    // workaround for the only direct use in HPX itself
    template <
        typename Tag1, typename Tag2, typename Tag3,
        typename T1, typename T2, typename T3>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_tuple<
        typename detail::tagged_type<Tag1, T1>::type,
        typename detail::tagged_type<Tag2, T2>::type,
        typename detail::tagged_type<Tag3, T3>::type
    >
    make_tagged_tuple(T1 && t1, T2 && t2, T3 && t3)
    {
        typedef tagged_tuple<
                typename detail::tagged_type<Tag1, T1>::type,
                typename detail::tagged_type<Tag2, T2>::type,
                typename detail::tagged_type<Tag3, T3>::type
            > result_type;

        return result_type(std::forward<T1>(t1), std::forward<T2>(t2),
            std::forward<T3>(t3));
    }

    template <
        typename Tag1, typename Tag2, typename Tag3,
        typename T1, typename T2, typename T3>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_tuple<
        typename detail::tagged_type<Tag1, T1>::type,
        typename detail::tagged_type<Tag2, T2>::type,
        typename detail::tagged_type<Tag3, T3>::type
    >
    make_tagged_tuple(tuple<T1, T2, T3> && t)
    {
        typedef tagged_tuple<
                typename detail::tagged_type<Tag1, T1>::type,
                typename detail::tagged_type<Tag2, T2>::type,
                typename detail::tagged_type<Tag3, T3>::type
            > result_type;

        return result_type(std::move(t));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Tag, std::size_t I, typename Tuple>
        struct tagged_element_type
        {
            typedef typename tuple_element<I, Tuple>::type element_type;
            typedef typename hpx::util::identity<Tag(element_type)>::type type;
        };

        template <typename Tuple, typename Indicies, typename ...Tags>
        struct tagged_tuple_helper;

        template <typename ...Ts, std::size_t ...Is, typename ...Tags>
        struct tagged_tuple_helper<
            tuple<Ts...>, pack_c<std::size_t, Is...>, Tags...>
        {
            typedef tagged_tuple<
                typename tagged_element_type<Tags, Is, tuple<Ts...> >::type...
            > type;
        };

        // workaround for broken MSVC2013
        template <typename ...Ts>
        struct pack_size
        {
            static const std::size_t value = sizeof...(Ts);
        };
    }

#if defined(HPX_HAVE_CXX11_EXPLICIT_VARIADIC_TEMPLATES)
    template <typename ...Tags, typename ...Ts>
    hpx::future<typename detail::tagged_tuple_helper<
        tuple<Ts...>,
        typename detail::make_index_pack<
            detail::pack_size<Tags...>::value
        >::type,
        Tags...
    >::type>
    make_tagged_tuple(hpx::future<tuple<Ts...> > && f)
    {
        static_assert(sizeof...(Tags) == tuple_size<tuple<Ts...> >::value,
            "the number of tags must be identical to the size of the given tuple");

        typedef typename detail::tagged_tuple_helper<
            tuple<Ts...>,
            typename detail::make_index_pack<sizeof...(Tags)>::type,
            Tags...
        >::type result_type;

        return lcos::make_future<result_type>(std::move(f),
            [](tuple<Ts...> && t) -> result_type
            {
                return make_tagged_tuple<Tags...>(std::move(t));
            });
    }
#else
    // workaround for the only direct use in HPX itself
    template <
        typename Tag1, typename Tag2, typename Tag3,
        typename T1, typename T2, typename T3>
    hpx::future<typename detail::tagged_tuple_helper<
        tuple<T1, T2, T3>,
        typename detail::make_index_pack<3ul>::type,
        Tag1, Tag2, Tag3
    >::type>
    make_tagged_tuple(hpx::future<tuple<T1, T2, T3> > && f)
    {
        typedef typename detail::tagged_tuple_helper<
            tuple<T1, T2, T3>,
            typename detail::make_index_pack<3ul>::type,
            Tag1, Tag2, Tag3
        >::type result_type;

        return lcos::make_future<result_type>(std::move(f),
            [](tuple<T1, T2, T3> && t) -> result_type
            {
                return make_tagged_tuple<Tag1, Tag2, Tag3>(std::move(t));
            });
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename ...Ts>
    struct tuple_size<tagged_tuple<Ts...> >
      : tuple_size<tuple<typename detail::tag_elem<Ts>::type...> >
    {};

    template <std::size_t N, typename ...Ts>
    struct tuple_element<N, tagged_tuple<Ts...> >
      : tuple_element<N, tuple<typename detail::tag_elem<Ts>::type...> >
    {};
}}

#endif
