//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/guided_chunk_size.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace hpx { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Iterations are dynamically assigned to threads in blocks as threads
    /// request them until no blocks remain to be assigned. Similar to
    /// \a dynamic_chunk_size except that the block size decreases each time a
    /// number of loop iterations is given to a thread. The size of the initial
    /// block is proportional to \a number_of_iterations / \a number_of_cores.
    /// Subsequent blocks are proportional to
    /// \a number_of_iterations_remaining / \a number_of_cores. The optional
    /// chunk size parameter defines the minimum block size. The default chunk
    /// size is 1.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's GUIDED
    ///       scheduling directive.
    ///
    struct guided_chunk_size
    {
        /// Construct a \a guided_chunk_size executor parameters object
        ///
        /// \param min_chunk_size [in] The optional minimal chunk size to use
        ///                     as the minimal number of loop iterations to
        ///                     schedule together.
        ///                     The default minimal chunk size is 1.
        ///
        constexpr explicit guided_chunk_size(std::size_t min_chunk_size = 1)
          : min_chunk_size_(min_chunk_size)
        {
        }

        /// \cond NOINTERNAL
        // This executor parameters type provides variable chunk sizes and
        // needs to be invoked for each of the chunks to be combined.
        typedef std::true_type has_variable_chunk_size;

        //         template <typename Executor>
        //         static std::size_t get_maximal_number_of_chunks(
        //             Executor && exec, std::size_t cores, std:size_t num_tasks)
        //         {
        //             // FIXME: find appropriate approximation
        //             return ...;
        //         }

        template <typename Executor, typename F>
        constexpr std::size_t get_chunk_size(Executor&& /* exec */, F&&,
            std::size_t cores, std::size_t num_tasks) const
        {
            return (std::max)(min_chunk_size_, (num_tasks + cores - 1) / cores);
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            ar& min_chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t min_chunk_size_;
        /// \endcond
    };
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {
    using guided_chunk_size HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::guided_chunk_size is deprecated. Use "
        "hpx::execution::guided_chunk_size instead.") =
        hpx::execution::guided_chunk_size;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<hpx::execution::guided_chunk_size>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
