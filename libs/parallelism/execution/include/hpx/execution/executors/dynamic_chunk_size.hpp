//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/dynamic_chunk_size.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    /// Loop iterations are divided into pieces of size \a chunk_size and then
    /// dynamically scheduled among the threads; when a thread finishes one
    /// chunk, it is dynamically assigned another If \a chunk_size is not
    /// specified, the default chunk size is 1.
    ///
    /// \note This executor parameters type is equivalent to OpenMP's DYNAMIC
    ///       scheduling directive.
    ///
    struct dynamic_chunk_size
    {
        /// Construct a \a dynamic_chunk_size executor parameters object
        ///
        /// \param chunk_size   [in] The optional chunk size to use as the
        ///                     number of loop iterations to schedule together.
        ///                     The default chunk size is 1.
        ///
        constexpr explicit dynamic_chunk_size(std::size_t chunk_size = 1)
          : chunk_size_(chunk_size)
        {
        }

        /// \cond NOINTERNAL
        template <typename Executor, typename F>
        constexpr std::size_t get_chunk_size(
            Executor&, F&&, std::size_t, std::size_t) const
        {
            return chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            ar& chunk_size_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t chunk_size_;
        /// \endcond
    };
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {
    using dynamic_chunk_size HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::dynamic_chunk_size is deprecated. Use "
        "hpx::execution::dynamic_chunk_size instead.") =
        hpx::execution::dynamic_chunk_size;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<hpx::execution::dynamic_chunk_size>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
