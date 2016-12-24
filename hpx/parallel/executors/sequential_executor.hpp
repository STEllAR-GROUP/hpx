//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM)
#define HPX_PARALLEL_EXECUTORS_SEQUENTIAL_EXECUTOR_MAY_11_2015_1050AM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/is_executor_v1.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/unwrapped.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

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

        template <typename F, typename ... Ts>
        static void apply_execute(F && f, Ts &&... ts)
        {
            execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        static typename hpx::util::detail::deferred_result_of<F(Ts&&...)>::type
        execute(F && f, Ts &&... ts)
        {
            try {
                return hpx::util::invoke(f, std::forward<Ts>(ts)...);
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

        template <typename F, typename ... Ts>
        static hpx::future<typename hpx::util::result_of<F&&(Ts&&...)>::type>
        async_execute(F && f, Ts &&... ts)
        {
            return hpx::async(launch::deferred, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename F, typename Shape, typename ... Ts>
        static std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, Shape, Ts...>::type
        > >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            typedef typename
                    detail::bulk_async_execute_result<F, Shape, Ts...>::type
                result_type;
            std::vector<hpx::future<result_type> > results;

            try {
                for (auto const& elem: shape)
                {
                    results.push_back(hpx::async(
                        launch::deferred, f, elem, ts...));
                }
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

        template <typename F, typename Shape, typename ... Ts>
        static typename detail::bulk_execute_result<F, Shape, Ts...>::type
        bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            return hpx::util::unwrapped(
                bulk_async_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...));
        }

        std::size_t processing_units_count()
        {
            return 1;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
        }
        /// \endcond
    };
}}}

#endif
