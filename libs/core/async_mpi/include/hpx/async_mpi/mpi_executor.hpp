//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_mpi/mpi_future.hpp>
#include <hpx/execution/executors/default_parameters.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/modules/mpi_base.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::mpi::experimental {

    struct executor
    {
        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = hpx::execution::parallel_execution_tag;

        // default params type as we don't do anything special
        using executor_parameters_type =
            hpx::execution::experimental::default_parameters;

        constexpr explicit executor(MPI_Comm communicator = MPI_COMM_WORLD)
          : communicator_(communicator)
        {
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(executor const& rhs) const noexcept
        {
            return communicator_ == rhs.communicator_;
        }

        constexpr bool operator!=(executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        constexpr executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t, executor const& exec,
            F&& f, Ts&&... ts)
        {
            return hpx::mpi::experimental::detail::async(
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)..., exec.communicator_);
        }

        std::size_t in_flight_estimate() const
        {
            return detail::get_mpi_info().requests_vector_size_ +
                detail::get_mpi_info().requests_queue_size_;
        }

    private:
        MPI_Comm communicator_;
    };
}    // namespace hpx::mpi::experimental
// namespace hpx::mpi::experimental

namespace hpx::parallel::execution {

    /// \cond NOINTERNAL
    template <>
    struct is_two_way_executor<hpx::mpi::experimental::executor>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
// namespace hpx::parallel::execution
