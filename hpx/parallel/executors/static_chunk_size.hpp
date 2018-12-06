//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/static_chunk_size.hpp

#if !defined(HPX_PARALLEL_STATIC_CHUNK_SIZE_JUL_31_2015_0740PM)
#define HPX_PARALLEL_STATIC_CHUNK_SIZE_JUL_31_2015_0740PM

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_executor_parameters.hpp>

#include <hpx/parallel/executors/execution_parameters_fwd.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces of size \a chunk_size and then
    /// assigned to threads. If \a chunk_size is not specified, the iterations
    /// are evenly (if possible) divided contiguously among the threads.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's STATIC
    ///       scheduling directive.
    ///
    struct static_chunk_size
    {
        /// Construct a \a static_chunk_size executor parameters object
        ///
        /// \note By default the number of loop iterations is determined from
        ///       the number of available cores and the overall number of loop
        ///       iterations to schedule.
        ///
        HPX_CONSTEXPR static_chunk_size()
          : chunk_size_(0)
        {}

        /// Construct a \a static_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to run on a single
        ///                     thread.
        ///
        HPX_CONSTEXPR explicit static_chunk_size(std::size_t chunk_size)
          : chunk_size_(chunk_size)
        {}

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        std::size_t get_chunk_size(Executor& exec, F &&, std::size_t cores,
            std::size_t num_tasks)
        {
            // use the given chunk size if given
            if (chunk_size_ != 0)
                return chunk_size_;

            // Make sure the internal round robin counter of the executor is
            // reset
            execution::reset_thread_distribution(*this, exec);

            // by default use static work distribution over number of
            // available compute resources, create four times the number of
            // chunks than we have cores
            return (num_tasks + 4 * cores - 1) / (4 * cores);   // round up
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t chunk_size_;
        /// \endcond
    };
}}}

namespace hpx { namespace parallel { namespace execution
{
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<parallel::execution::static_chunk_size>
      : std::true_type
    {};
    /// \endcond
}}}

#endif
