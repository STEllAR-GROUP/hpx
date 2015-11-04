//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM)
#define HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM

#include <hpx/config.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unwrapped.hpp>

#include <type_traits>
#include <utility>
#include <iterator>

#include <boost/range/functions.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/type_traits/is_void.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// A \a sequential_executor creates groups of sequential execution agents
    /// which execute in the calling thread. The sequential order is given by
    /// the lexicographical order of indices in the index space.
    ///
    struct sequential_executor : executor_tag
    {
#if defined(DOXYGEN)
        /// Create a new sequential executor
        sequential_executor() {}
#endif

        /// \cond NOINTERNAL
        typedef sequential_execution_tag execution_category;

        template <typename F>
        static void apply_execute(F && f)
        {
            execute(std::forward<F>(f));
        }

        template <typename F>
        static typename hpx::util::result_of<
            typename hpx::util::decay<F>::type()
        >::type
        execute(F && f)
        {
            try {
                return f();
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                boost::throw_exception(
                    exception_list(boost::current_exception())
                );
            }
        }

        template <typename F>
        static hpx::future<typename hpx::util::result_of<
            typename hpx::util::decay<F>::type()
        >::type>
        async_execute(F && f)
        {
            return hpx::async(launch::deferred, std::forward<F>(f));
        }

        template <typename F, typename Shape>
        static std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, Shape>::type
        > >
        bulk_async_execute(F && f, Shape const& shape)
        {
            typedef typename
                    detail::bulk_async_execute_result<F, Shape>::type
                result_type;
            std::vector<hpx::future<result_type> > results;

            try {
                for (auto const& elem: shape)
                    results.push_back(hpx::async(launch::deferred, f, elem));
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                boost::throw_exception(
                    exception_list(boost::current_exception())
                );
            }

            return std::move(results);
        }

        template <typename F, typename Shape>
        static typename detail::bulk_execute_result<F, Shape>::type
        bulk_execute(F && f, Shape const& shape)
        {
            return hpx::util::unwrapped(
                bulk_async_execute(std::forward<F>(f), shape));
        }

        std::size_t processing_units_count()
        {
            return 1;
        }
        /// \endcond
    };
}}}

#endif
