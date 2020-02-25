//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MPI_EXECUTOR_HPP)
#define HPX_MPI_EXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution_fwd.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/mpi/mpi_future.hpp>

#include <mpi.h>

namespace hpx { namespace mpi {

    struct executor
    {
        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = parallel::execution::parallel_execution_tag;

        constexpr executor(MPI_Comm communicator = MPI_COMM_WORLD)
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
        decltype(auto) async_execute(F&& f, Ts&&... ts) const
        {
            return hpx::mpi::detail::async(
                std::forward<F>(f), std::forward<Ts>(ts)..., communicator_);
        }

    private:
        MPI_Comm communicator_;
    };
}}    // namespace hpx::mpi

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <>
    struct is_two_way_executor<hpx::mpi::executor> : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution

#endif
