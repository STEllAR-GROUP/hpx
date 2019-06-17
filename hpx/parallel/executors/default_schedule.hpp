
//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_DEFAULT_SCHEDULE)
#define HPX_PARALLEL_EXECUTORS_DEFAULT_SCHEDULE

#include <hpx/config.hpp>
#include <hpx/parallel/executors/chunked_placement.hpp>
#include <hpx/parallel/executors/execution_parameters_fwd.hpp>
#include <hpx/parallel/executors/static_chunk_size.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// This executor parameter represents the default loop schedule for bulk
    /// execution. Loop scheduling means 1. the chunking and 2. the placement of
    /// tasks. The default schedule uses static chunking and chunked placement
    /// which groups tasks with similar index on the same cores or pus. Note
    /// that this is *not* exactly the same as OpenMP's static schedule since
    /// the underlying thread pool may have work-stealing enabled.
    struct default_schedule
    {
        /// Construct a \a default_schedule executor parameters object.
        HPX_CONSTEXPR default_schedule() {}

        /// Construct a \a default_schedule executor parameters object.
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to run on a single
        ///                     thread.
        ///
        HPX_CONSTEXPR explicit default_schedule(std::size_t chunk_size)
          : chunker_(chunk_size)
        {
        }

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        std::size_t get_chunk_size(
            Executor& exec, F&& f, std::size_t cores, std::size_t num_tasks)
        {
            return chunker_.get_chunk_size(
                exec, std::forward<F>(f), cores, num_tasks);
        }

        template <typename Executor>
        hpx::threads::thread_schedule_hint get_schedule_hint(Executor& exec,
            std::size_t task_idx, std::size_t num_tasks, std::size_t cores)
        {
            return schedule_.get_schedule_hint(
                exec, task_idx, num_tasks, cores);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & schedule_;
            ar & chunker_;
        }

        chunked_placement schedule_;
        static_chunk_size chunker_;
        /// \endcond
    };
}}}

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<parallel::execution::default_schedule>
      : std::true_type
    {
    };
    /// \endcond
}}}

#endif
