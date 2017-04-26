//  Copyright Eric Niebler 2013-2015
//  Copyright 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This was modeled after the code available in the Range v3 library

#if !defined(HPX_UTIL_TAGGED_PAIR_DEC_23_2015_1014AM)
#define HPX_UTIL_TAGGED_PAIR_DEC_23_2015_1014AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/tagged.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename S>
    struct tagged_pair
      : tagged<
            std::pair<
                typename detail::tag_elem<F>::type,
                typename detail::tag_elem<S>::type
            >,
            typename detail::tag_spec<F>::type,
            typename detail::tag_spec<S>::type
        >
    {
        typedef tagged<
                std::pair<
                    typename detail::tag_elem<F>::type,
                    typename detail::tag_elem<S>::type
                >,
                typename detail::tag_spec<F>::type,
                typename detail::tag_spec<S>::type
            > base_type;

        template <typename ... Ts>
        tagged_pair(Ts && ... ts)
          : base_type(std::forward<Ts>(ts)...)
        {}
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_pair<Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)>
    make_tagged_pair(std::pair<T1, T2> && p)
    {
        typedef tagged_pair<
                Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)
            > result_type;

        return result_type(std::move(p));
    }

    template <typename Tag1, typename Tag2, typename T1, typename T2>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_pair<Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)>
    make_tagged_pair(std::pair<T1, T2> const& p)
    {
        typedef tagged_pair<
                Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)
            > result_type;

        return result_type(p);
    }

    template <typename Tag1, typename Tag2, typename ... Ts>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_pair<
        Tag1(typename tuple_element<0, tuple<Ts...> >::type),
        Tag2(typename tuple_element<1, tuple<Ts...> >::type)
    >
    make_tagged_pair(tuple<Ts...> && p)
    {
        static_assert(sizeof...(Ts) >= 2, "tuple must have at least 2 elements");

        typedef tagged_pair<
                Tag1(typename tuple_element<0, tuple<Ts...> >::type),
                Tag2(typename tuple_element<1, tuple<Ts...> >::type)
            > result_type;

        return result_type(std::move(get<0>(p)), std::move(get<1>(p)));
    }

    template <typename Tag1, typename Tag2, typename ... Ts>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_pair<
        Tag1(typename tuple_element<0, tuple<Ts...> >::type),
        Tag2(typename tuple_element<1, tuple<Ts...> >::type)
    >
    make_tagged_pair(tuple<Ts...> const& p)
    {
        static_assert(sizeof...(Ts) >= 2, "tuple must have at least 2 elements");

        typedef tagged_pair<
                Tag1(typename tuple_element<0, tuple<Ts...> >::type),
                Tag2(typename tuple_element<1, tuple<Ts...> >::type)
            > result_type;

        return result_type(get<0>(p), get<1>(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    hpx::future<tagged_pair<
        Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)
    > >
    make_tagged_pair(hpx::future<std::pair<T1, T2> > && f)
    {
        typedef hpx::util::tagged_pair<
                Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)
            > result_type;

        return lcos::make_future<result_type>(std::move(f),
            [](std::pair<T1, T2> && p) -> result_type
            {
                return make_tagged_pair<Tag1, Tag2>(std::move(p));
            });
    }

    template <typename Tag1, typename Tag2, typename ... Ts>
    hpx::future<tagged_pair<
        Tag1(typename tuple_element<0, tuple<Ts...> >::type),
        Tag2(typename tuple_element<1, tuple<Ts...> >::type)
    > >
    make_tagged_pair(hpx::future<tuple<Ts...> > && f)
    {
        static_assert(sizeof...(Ts) >= 2, "tuple must have at least 2 elements");

        typedef hpx::util::tagged_pair<
                Tag1(typename tuple_element<0, tuple<Ts...> >::type),
                Tag2(typename tuple_element<1, tuple<Ts...> >::type)
            > result_type;

        return lcos::make_future<result_type>(std::move(f),
            [](tuple<Ts...> && p) -> result_type
            {
                return make_tagged_pair<Tag1, Tag2>(std::move(p));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename T1, typename T2>
    HPX_CONSTEXPR HPX_FORCEINLINE
    tagged_pair<Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)>
    make_tagged_pair(T1 && t1, T2 && t2)
    {
        typedef tagged_pair<
                Tag1(typename decay<T1>::type), Tag2(typename decay<T2>::type)
            > result_type;

        return result_type(std::forward<T1>(t1), std::forward<T2>(t2));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2>
    struct tuple_size<tagged_pair<Tag1, Tag2> >
      : std::integral_constant<std::size_t, 2>
    {};

    template <std::size_t N, typename Tag1, typename Tag2>
    struct tuple_element<N, tagged_pair<Tag1, Tag2> >
      : tuple_element<N, std::pair<
            typename detail::tag_elem<Tag1>::type,
            typename detail::tag_elem<Tag2>::type
        > >
    {};
}}

#endif
