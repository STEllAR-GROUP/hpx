//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_UTIL_PARTITIONER_MAY_27_2014_1040PM)
#define HPX_STL_UTIL_PARTITIONER_MAY_27_2014_1040PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/stl/execution_policy.hpp>
#include <hpx/stl/detail/algorithm_result.hpp>

namespace hpx { namespace parallel { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct static_partitioner_tag {};
    struct default_partitioner_tag {};

    namespace detail
    {
        template <typename ExPolicy>
        struct extract_partitioner
        {
            typedef default_partitioner_tag type;
        };

        ///////////////////////////////////////////////////////////////////////
        // std::bad_alloc has to be handled separately
        void handle_exception(boost::exception_ptr const& e,
            std::list<boost::exception_ptr>& errors)
        {
            try {
                boost::rethrow_exception(e);
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                errors.push_back(e);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy,
        typename PartTag = typename detail::extract_partitioner<ExPolicy>::type>
    struct partitioner;

    template <typename ExPolicy>
    struct partitioner<ExPolicy, static_partitioner_tag>
    {
        template <typename FwdIter, typename F>
        static FwdIter call(FwdIter first, std::size_t count, F && func,
            std::size_t chunk_size = 0)
        {
            // estimate a chunk size based on number of cores used
            if (chunk_size == 0)
            {
                std::size_t const cores = hpx::get_num_worker_threads();
                chunk_size = (count + cores - 1) / cores;
            }

            // schedule every chunk on a separate thread
            std::vector<hpx::future<void> > workitems;
            workitems.reserve(count / chunk_size + 1);

            while (count > chunk_size)
            {
                workitems.emplace_back(hpx::async(
                    hpx::util::bind(func, first, chunk_size)));
                count -= chunk_size;
                std::advance(first, chunk_size);
            }

            std::list<boost::exception_ptr> errors;

            // execute last chunk directly
            if (count != 0)
            {
                // std::bad_alloc has to be handled separately
                try {
                    func(first, count);
                }
                catch (std::bad_alloc const& e) {
                    boost::throw_exception(e);
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }
                std::advance(first, count);
            }

            // wait for all tasks to finish
            hpx::wait_all(workitems);

            for (hpx::future<void>& f: workitems)
            {
                if (f.has_exception())
                    detail::handle_exception(f.get_exception_ptr(), errors);
            }

            if (!errors.empty())
                boost::throw_exception(exception_list(std::move(errors)));

            return first;
        }
    };

    template <>
    struct partitioner<task_execution_policy, static_partitioner_tag>
    {
        template <typename FwdIter, typename F>
        static hpx::future<FwdIter> call(FwdIter first, std::size_t count,
            F && func, std::size_t chunk_size = 0)
        {
            // estimate a chunk size based on number of cores used
            if (chunk_size == 0)
            {
                std::size_t const cores = hpx::get_num_worker_threads();
                chunk_size = (count + cores - 1) / cores;
            }

            // schedule every chunk on a separate thread
            std::vector<hpx::future<void> > workitems;
            workitems.reserve(count / chunk_size + 1);

            while (count > chunk_size)
            {
                workitems.emplace_back(hpx::async(
                    hpx::util::bind(func, first, chunk_size)));
                count -= chunk_size;
                std::advance(first, chunk_size);
            }

            // add last chunk
            if (count != 0)
            {
                workitems.emplace_back(hpx::async(
                    hpx::util::bind(func, first, count)));
                std::advance(first, count);
            }

            // wait for all tasks to finish
            return hpx::when_all(workitems).then(
                [first](hpx::future<std::vector<hpx::future<void> > >&& r)
                {
                    std::vector<hpx::future<void> > result = r.get();

                    std::list<boost::exception_ptr> errors;
                    for (hpx::future<void>& f: result)
                    {
                        if (f.has_exception())
                            detail::handle_exception(f.get_exception_ptr(), errors);
                    }

                    if (!errors.empty())
                        boost::throw_exception(exception_list(std::move(errors)));

                    return first;
                }
            );
        }
    };

    template <typename ExPolicy>
    struct partitioner<ExPolicy, default_partitioner_tag>
      : partitioner<ExPolicy, static_partitioner_tag>
    {};
}}}

#endif
