//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_UTIL_PARTITIONER_MAY_27_2014_1040PM)
#define HPX_STL_UTIL_PARTITIONER_MAY_27_2014_1040PM

#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace parallale { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct static_partitioner_tag {};
    struct default_partitioner_tag {};

    template <typename PartTag>
    struct partitioner;

    template <>
    struct partitioner<static_partitioner_tag>
    {
        template <typename FwdIter, typename Diff, typename F>
        static FwdIter call(FwdIter first, Diff count, F && func,
            std::size_t chunk_size = 0)
        {
            // estimate a chunk size based on number of cores used
            if (chunk_size == 0)
            {
                unsigned int const _HdConc = std::thread::hardware_concurrency();
                chunk_size = (count + _HdConc - 1) / _HdConc;
            }

            // schedule every chunk on a separate thread
            std::vector<hpx::future<void> > workitems;
            workitems.reserve(count / chunk_size + 1);

            while (count > chunk_size)
            {
                workitems.emplace_back(hpx::async(util::bind(func, first, chunk_size)));
                count -= chunk_size;
                std::advance(first, chunk_size);
            }

            // execute last chunk directly
            func(first, count);
            std::advance(first, count);

            // wait for all tasks to finish
            hpx::wait_all(workitems);
            return first;
        }
    };
}}}

#endif
