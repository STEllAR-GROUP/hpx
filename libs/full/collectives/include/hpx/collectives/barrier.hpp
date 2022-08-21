//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/collectives/barrier.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/memory.hpp>

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::distributed {

    /// The barrier is an implementation performing a barrier over a number of
    /// participating threads. The different threads don't have to be on the
    /// same locality. This barrier can be invoked in a distributed application.
    ///
    /// For a local only barrier \see hpx::barrier.
    class HPX_EXPORT barrier
    {
    public:
        /// Creates a barrier, rank is locality id, size is number of localities
        ///
        /// \param base_name The name of the barrier
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the created communictator
        ///
        /// A barrier \a base_name is created. It expects that
        /// hpx::get_num_localities() participate and the local rank is
        /// hpx::get_locality_id().

        explicit barrier(std::string const& base_name,
            hpx::collectives::generation_arg generation = {},
            hpx::collectives::root_site_arg root_site = {});

        /// Creates a barrier with a given size, rank is locality id
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the created communictator
        ///
        /// A barrier \a base_name is created. It expects that \a num
        /// participate and the local rank is hpx::get_locality_id().
        ///
        barrier(std::string const& base_name,
            hpx::collectives::num_sites_arg num,
            hpx::collectives::generation_arg generation = {},
            hpx::collectives::root_site_arg root_site = {});

        HPX_DEPRECATED_V(1, 9,
            "this barrier::barrier constructor is deprecated, use the "
            "constructor taking a num_sites_arg instead")
        barrier(std::string const& base_name, std::size_t num)
          : barrier(base_name, hpx::collectives::num_sites_arg(num))
        {
        }

        /// Creates a barrier with a given size and rank
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        /// \param rank The rank of the calling site for this invocation
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the created communictator
        ///
        /// A barrier \a base_name is created. It expects that
        /// \a num participate and the local rank is \a rank.
        ///
        barrier(std::string const& base_name,
            hpx::collectives::num_sites_arg num,
            hpx::collectives::this_site_arg rank,
            hpx::collectives::generation_arg generation = {},
            hpx::collectives::root_site_arg root_site = {});

        HPX_DEPRECATED_V(1, 9,
            "this barrier::barrier constructor is deprecated, use the "
            "constructor taking a num_sites_arg and a this_site_arg instead")
        barrier(std::string const& base_name, std::size_t num, std::size_t rank)
          : barrier(base_name, hpx::collectives::num_sites_arg(num),
                hpx::collectives::this_site_arg(rank))
        {
        }

        /// Creates a barrier with a vector of ranks
        ///
        /// \param base_name The name of the barrier
        ///
        /// \param ranks Gives a list of participating ranks (this could be
        ///              derived from a list of locality ids
        /// \param rank The rank of the calling site for this invocation
        ///
        /// A barrier \a base_name is created. It expects that ranks.size() and
        /// the local rank is \a rank (must be contained in \a ranks).
        ///
        /// \note This constructor is deprecated and will be removed in the
        ///       future.
        ///
        HPX_DEPRECATED_V(
            1, 9, "this barrier::barrier constructor is deprecated")
        barrier(std::string const& base_name,
            std::vector<std::size_t> const& ranks, std::size_t rank);

        /// \cond NOINTERNAL
        barrier(barrier&& other) noexcept;
        barrier& operator=(barrier&& other) noexcept;

        /// \cond NOINTERNAL
        ~barrier();
        /// \endcond

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        ///
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the barrier operation.
        ///
        /// \returns This function returns once all participants have entered
        /// the barrier (have called \a wait).
        void wait(hpx::collectives::generation_arg generation = {}) const;

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        ///
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the barrier operation.
        ///
        /// \returns A future that becomes ready once all participants have
        ///          entered the barrier (have called \a wait).
        ///
        hpx::future<void> wait(hpx::launch::async_policy,
            hpx::collectives::generation_arg generation = {});

        /// \cond NOINTERNAL
        // Get the instance of the global barrier
        static barrier& get_global_barrier();

        // detach the communicator
        void detach();
        /// \endcond

    public:
        /// Perform a global synchronization using the default global barrier
        /// The barrier is created once at startup and can be reused throughout
        /// the lifetime of an HPX application.
        ///
        /// \param generation An optional generational sequence number used to
        ///                   distinguish the barrier operation.
        ///
        /// \note This function currently does not support dynamic connection
        ///       and disconnection of localities.
        ///
        static void synchronize(
            hpx::collectives::generation_arg generation = {});

    private:
        /// \cond NOINTERNAL
        barrier();

        std::atomic<std::size_t> generation_{0};
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        hpx::collectives::communicator comm_;
#endif
        /// \endcond
    };
}    // namespace hpx::distributed

/// \cond NOINTERNAL
namespace hpx::lcos {

    using barrier HPX_DEPRECATED_V(1, 8,
        "hpx::lcos::barrier is deprecated, use hpx::distributed::barrier "
        "instead") = hpx::distributed::barrier;
}
/// \endcond

#include <hpx/config/warnings_suffix.hpp>
