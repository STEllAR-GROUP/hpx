//  Copyright (c) 2022 Hartmut Kaiser
//  Copyright (c) 2022 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/numcores.hpp

#pragma once

#include <hpx/execution/executors/execution_parameters_fwd.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx::execution::experimental {

    /// Control number of cores in executors which need a functionality
    /// for setting the number of cores to be used by an algorithm directly
    ///
    struct num_cores
    {
        /// Construct a \a num_cores executor parameters object
        ///
        /// \note make sure the minimal number of cores is 1
        ///
        constexpr explicit num_cores(std::size_t cores = 1) noexcept
          : num_cores_(cores == 0 ? 1 : cores)
        {
        }

        /// \cond NOINTERNAL
        // discover the number of cores to use for parallelization
        template <typename Executor>
        constexpr std::size_t processing_units_count(Executor&&,
            hpx::chrono::steady_duration const&, std::size_t) const noexcept
        {
            return num_cores_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & num_cores_;
            // clang-format on
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        std::size_t num_cores_;
        /// \endcond
    };
}    // namespace hpx::execution::experimental

namespace hpx::parallel::execution {

    /// \cond NOINTERNAL
    template <>
    struct is_executor_parameters<hpx::execution::experimental::num_cores>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution

namespace hpx::execution {

    using num_cores HPX_DEPRECATED_V(1, 9,
        "hpx::execution::num_cores is deprecated, use "
        "hpx::execution::experimental::num_cores instead") =
        hpx::execution::experimental::num_cores;
}
