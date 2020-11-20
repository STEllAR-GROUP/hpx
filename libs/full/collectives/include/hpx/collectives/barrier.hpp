//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/collectives/barrier.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/collectives/detail/barrier_node.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace lcos {

    /// \cond NOINTERNAL
    namespace detail {

        struct barrier_node;
    }
    /// \endcond

    /// The barrier is an implementation performing a barrier over a number of
    /// participating threads. The different threads don't have to be on the
    /// same locality. This barrier can be invoked in a distributed application.
    ///
    /// For a local only barrier \see hpx::lcos::local::barrier.
    class HPX_EXPORT barrier
    {
        /// \cond NOINTERNAL
        typedef detail::barrier_node wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;
        /// \endcond

    public:
        /// Creates a barrier, rank is locality id, size is number of localities
        ///
        /// \param base_name The name of the barrier
        ///
        /// A barrier \a base_name is created. It expects that
        /// hpx::get_num_localities() participate and the local rank is
        /// hpx::get_locality_id().
        barrier(std::string const& base_name);

        /// Creates a barrier with a given size, rank is locality id
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        ///
        /// A barrier \a base_name is created. It expects that
        /// \a num participate and the local rank is hpx::get_locality_id().
        barrier(std::string const& base_name, std::size_t num);

        /// Creates a barrier with a given size and rank
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        /// \param rank The rank of the calling site for this invocation
        ///
        /// A barrier \a base_name is created. It expects that
        /// \a num participate and the local rank is \a rank.
        barrier(
            std::string const& base_name, std::size_t num, std::size_t rank);

        /// Creates a barrier with a vector of ranks
        ///
        /// \param base_name The name of the barrier
        /// \param ranks Gives a list of participating ranks (this could be derived
        ///              from a list of locality ids
        /// \param rank The rank of the calling site for this invocation
        ///
        /// A barrier \a base_name is created. It expects that ranks.size()
        /// and the local rank is \a rank (must be contained in \a ranks).
        barrier(std::string const& base_name,
            std::vector<std::size_t> const& ranks, std::size_t rank);

        /// \cond NOINTERNAL
        barrier(barrier&& other);
        barrier& operator=(barrier&& other);

        /// \cond NOINTERNAL
        ~barrier();
        /// \endcond

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        ///
        /// \returns This function returns once all participants have entered
        /// the barrier (have called \a wait).
        void wait();

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        ///
        /// \returns a future that becomes ready once all participants have
        /// entered the barrier (have called \a wait).
        hpx::future<void> wait(hpx::launch::async_policy);

        /// \cond NOINTERNAL
        // Resets this barrier instance.
        void release();

        void detach();

        // Get the instance of the global barrier
        static barrier& get_global_barrier();
        static barrier create_global_barrier();
        /// \endcond

        /// Perform a global synchronization using the default global barrier
        /// The barrier is created once at startup and can be reused throughout
        /// the lifetime of an HPX application.
        ///
        /// \note This function currently does not support dynamic connection
        /// and disconnection of localities.
        static void synchronize();

    private:
        /// \cond NOINTERNAL
        barrier();

        hpx::intrusive_ptr<wrapping_type> node_;
        /// \endcond
    };
}}    // namespace hpx::lcos

#include <hpx/config/warnings_suffix.hpp>
