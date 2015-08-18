//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM)
#define HPX_PARALLEL_UTIL_ZIP_ITERATOR_MAY_29_2014_0852PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/zip_iterator.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1) { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <int N, typename R, typename ZipIter>
    R get_iter(ZipIter&& zipiter)
    {
        return hpx::util::get<N>(zipiter.get_iterator_tuple());
    }

    template <int N, typename R, typename ZipIter>
    R get_iter(hpx::future<ZipIter>&& zipiter)
    {
        typedef typename hpx::util::tuple_element<
            N, typename ZipIter::iterator_tuple_type
        >::type result_type;

        return zipiter.then(
            [](hpx::future<ZipIter>&& f) -> result_type
            {
                return hpx::util::get<N>(f.get().get_iterator_tuple());
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename ZipIter>
    R get_iter_tuple(ZipIter&& zipiter)
    {
        return zipiter.get_iterator_tuple();
    }

    template <typename R, typename ZipIter>
    R get_iter_tuple(hpx::future<ZipIter>&& zipiter)
    {
        typedef typename ZipIter::iterator_tuple_type result_type;

        return zipiter.then(
            [](hpx::future<ZipIter>&& f) -> result_type
            {
                return f.get().get_iterator_tuple();
            });
    }
}}}}

#endif
