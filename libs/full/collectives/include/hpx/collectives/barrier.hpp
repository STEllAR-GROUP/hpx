//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/collectives/barrier.hpp
/// \page hpx::distributed::barrier
/// \headerfile hpx/barrier.hpp

#pragma once

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace distributed {

    /// The barrier is an implementation performing a barrier over a number of
    /// participating threads. The different threads don't have to be on the
    /// same locality. This barrier can be invoked in a distributed application.
    ///
    /// For a local only barrier \see hpx::barrier.
    class HPX_EXPORT barrier;

    /// Creates a barrier, rank is locality id, size is number of localities
    ///
    /// \param base_name The name of the barrier
    ///
    /// A barrier \a base_name is created. It expects that
    /// hpx::get_num_localities() participate and the local rank is
    /// hpx::get_locality_id().
    explicit barrier(std::string const& base_name);

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

    /// Wait until each participant entered the barrier. Must be called by
    /// all participants
    ///
    /// \returns This function returns once all participants have entered
    /// the barrier (have called \a wait).
    void wait() const;

    /// Wait until each participant entered the barrier. Must be called by
    /// all participants
    ///
    /// \returns a future that becomes ready once all participants have
    /// entered the barrier (have called \a wait).
    hpx::future<void> wait(hpx::launch::async_policy) const;

    /// Perform a global synchronization using the default global barrier
    /// The barrier is created once at startup and can be reused throughout
    /// the lifetime of an HPX application.
    ///
    /// \note This function currently does not support dynamic connection
    /// and disconnection of localities.
    static void synchronize();

}}    // namespace hpx::distributed

// clang-format on
#else

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/collectives/detail/barrier_node.hpp>
#include <hpx/components_base/server/managed_component_base.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/memory.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::distributed {

    namespace detail {

        struct barrier_node;
    }

    class HPX_EXPORT barrier
    {
        typedef detail::barrier_node wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        explicit barrier(std::string const& base_name);

        barrier(std::string const& base_name, std::size_t num);

        barrier(
            std::string const& base_name, std::size_t num, std::size_t rank);

        barrier(std::string const& base_name,
            std::vector<std::size_t> const& ranks, std::size_t rank);

        barrier(barrier&& other) noexcept;
        barrier& operator=(barrier&& other) noexcept;

        ~barrier();

        void wait() const;

        hpx::future<void> wait(hpx::launch::async_policy) const;

        // Resets this barrier instance.
        void release();

        void detach();

        // Get the instance of the global barrier
        static std::array<barrier, 2>& get_global_barrier();
        static std::array<barrier, 2> create_global_barrier();

        static void synchronize();

    private:
        barrier();

        hpx::intrusive_ptr<wrapping_type> node_;
    };
}    // namespace hpx::distributed

#include <hpx/config/warnings_suffix.hpp>

#endif    // DOXYGEN
