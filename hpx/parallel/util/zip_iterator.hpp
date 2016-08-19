//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/tagged_pair.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/zip_iterator.hpp>

#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename R, typename ZipIter>
    R get_iter(ZipIter && zipiter)
    {
        return hpx::util::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(hpx::future<ZipIter> && zipiter)
    {
        typedef typename hpx::util::tuple_element<
            N, typename ZipIter::iterator_tuple_type
        >::type result_type;

        return lcos::make_future<result_type>(std::move(zipiter),
            [](ZipIter zipiter)
            {
                return get_iter<N, result_type>(std::move(zipiter));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    typename ZipIter::iterator_tuple_type
    get_iter_tuple(ZipIter && zipiter)
    {
        return zipiter.get_iterator_tuple();
    }

    template <typename ZipIter>
    hpx::future<typename ZipIter::iterator_tuple_type>
    get_iter_tuple(hpx::future<ZipIter> && zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type result_type;
        return lcos::make_future<result_type>(std::move(zipiter),
            [](ZipIter zipiter)
            {
                return get_iter_tuple(std::move(zipiter));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ZipIter>
    std::pair<
        typename hpx::util::tuple_element<
            0, typename ZipIter::iterator_tuple_type
        >::type,
        typename hpx::util::tuple_element<
            1, typename ZipIter::iterator_tuple_type
        >::type
    >
    get_iter_pair(ZipIter && zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type iterator_tuple_type;

        iterator_tuple_type const& t = zipiter.get_iterator_tuple();
        return std::make_pair(hpx::util::get<0>(t), hpx::util::get<1>(t));
    }

    template <typename ZipIter>
    hpx::future<std::pair<
        typename hpx::util::tuple_element<
            0, typename ZipIter::iterator_tuple_type
        >::type,
        typename hpx::util::tuple_element<
            1, typename ZipIter::iterator_tuple_type
        >::type
    > >
    get_iter_pair(hpx::future<ZipIter> && zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type iterator_tuple_type;

        typedef std::pair<
            typename hpx::util::tuple_element<0, iterator_tuple_type>::type,
            typename hpx::util::tuple_element<1, iterator_tuple_type>::type
        > result_type;

        return lcos::make_future<result_type>(std::move(zipiter),
            [](ZipIter zipiter)
            {
                return get_iter_pair(std::move(zipiter));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Tag1, typename Tag2, typename ZipIter>
    hpx::util::tagged_pair<
        Tag1(typename hpx::util::tuple_element<
            0, typename ZipIter::iterator_tuple_type
        >::type),
        Tag2(typename hpx::util::tuple_element<
            1, typename ZipIter::iterator_tuple_type
        >::type)
    >
    get_iter_tagged_pair(ZipIter && zipiter)
    {
        return hpx::util::make_tagged_pair<Tag1, Tag2>(
            zipiter.get_iterator_tuple());
    }

    template <typename Tag1, typename Tag2, typename ZipIter>
    hpx::future<hpx::util::tagged_pair<
        Tag1(typename hpx::util::tuple_element<
            0, typename ZipIter::iterator_tuple_type
        >::type),
        Tag2(typename hpx::util::tuple_element<
            1, typename ZipIter::iterator_tuple_type
        >::type)
    > >
    get_iter_tagged_pair(hpx::future<ZipIter> && zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type iterator_tuple_type;

        typedef hpx::util::tagged_pair<
            Tag1(typename hpx::util::tuple_element<0, iterator_tuple_type>::type),
            Tag2(typename hpx::util::tuple_element<1, iterator_tuple_type>::type)
        > result_type;

        return lcos::make_future<result_type>(std::move(zipiter),
            [](ZipIter && zipiter)
            {
                return get_iter_tagged_pair<Tag1, Tag2>(std::move(zipiter));
            });
    }
}}}}

#endif
