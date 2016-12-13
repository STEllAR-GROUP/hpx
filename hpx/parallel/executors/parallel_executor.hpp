//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/parallel_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM)
#define HPX_PARALLEL_EXECUTORS_PARALLEL_EXECUTOR_MAY_13_2015_1057AM

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/deferred_call.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/functions.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// A \a parallel_executor creates groups of parallel execution agents
    /// which execute in threads implicitly created by the executor. This
    /// executor prefers continuing with the creating thread first before
    /// executing newly created threads.
    struct parallel_executor : executor_tag
    {
        /// Associate the auto_chunk_size executor parameters type as a default
        /// with this executor.
        typedef static_chunk_size executor_parameters_type;

        /// Create a new parallel executor
        explicit parallel_executor(launch l = launch::async,
                std::size_t spread = 4, std::size_t tasks =
                    (std::min)(std::size_t(128), hpx::get_os_thread_count()))
          : l_(l), num_spread_(spread), num_tasks_(tasks)
        {}

        /// \cond NOINTERNAL
        template <typename F, typename ... Ts>
        static void apply_execute(F && f, Ts &&... ts)
        {
            hpx::apply(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<
            typename hpx::util::detail::deferred_result_of<F(Ts&&...)>::type
        >
        async_execute(F && f, Ts &&... ts) const
        {
            return hpx::async(l_, std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename S, typename ... Ts>
        std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, S, Ts...>::type
        > >
        bulk_async_execute(F && f, S const& shape, Ts &&... ts)
        {
            typedef std::vector<hpx::future<
                    typename detail::bulk_async_execute_result<F, S, Ts...>::type
                > > result_type;

            result_type results;

// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
            std::size_t size = std::distance(boost::begin(shape),
                boost::end(shape));
#else
            std::size_t size = boost::size(shape);
#endif

            results.resize(size);
            spawn(results, 0, size, f, boost::begin(shape), ts...).get();
            return results;
        }
        /// \endcond

    protected:
        /// \cond NOINTERNAL
        template <typename Result, typename F, typename Iter, typename ... Ts>
        hpx::future<void> spawn(std::vector<hpx::future<Result> >& results,
            std::size_t base, std::size_t size, F const& func, Iter it,
            Ts const&... ts)
        {
            if (size > num_tasks_)
            {
                // spawn hierarchical tasks
                std::size_t chunk_size = (size + num_spread_) / num_spread_ - 1;
                chunk_size = (std::max)(chunk_size, num_tasks_);

                std::vector<hpx::future<void> > tasks;
                tasks.reserve(num_spread_);

                hpx::future<void> (parallel_executor::*spawn_func)(
                        std::vector<hpx::future<Result> >&,
                        std::size_t, std::size_t, F const&, Iter, Ts const&...
                    ) = &parallel_executor::spawn;

                for (std::size_t i = 0; i != num_spread_ && size != 0; ++i)
                {
                    std::size_t curr_chunk_size = (std::min)(chunk_size, size);

                    hpx::future<void> f = hpx::async(
                        spawn_func, this, std::ref(results), base,
                        curr_chunk_size, std::ref(func), it, std::ref(ts)...);
                    tasks.push_back(std::move(f));

                    base += curr_chunk_size;
                    it = hpx::parallel::v1::detail::next(it, curr_chunk_size);
                    size -= curr_chunk_size;
                }

                HPX_ASSERT(size == 0);

                return hpx::when_all(tasks);
            }

            // spawn all tasks sequentially
            HPX_ASSERT(base + size <= results.size());

            for (std::size_t i = 0; i != size; ++i, ++it)
            {
                results[base + i] = hpx::async(l_, func, *it, ts...);
            }

            return hpx::make_ready_future();
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & l_ & num_spread_ & num_tasks_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        launch l_;
        std::size_t num_spread_;
        std::size_t num_tasks_;
        /// \endcond
    };
}}}

#endif
