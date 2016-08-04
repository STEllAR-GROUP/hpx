//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_BARRIER_HPP
#define HPX_LCOS_BARRIER_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/launch_policy.hpp>

#include <boost/intrusive_ptr.hpp>

#include <string>

namespace hpx { namespace lcos { namespace detail {
    struct HPX_EXPORT barrier_node;
}}}

namespace hpx { namespace lcos {
    /// \cond NOINTERNAL
    namespace detail
    {
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
        typedef detail::barrier_node wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
        /// Creates a barrier, rank is locality id, size is number of localities
        ///
        /// \param base_name The name of the barrier
        ///
        /// A barrier \param base_name is created. It expects that
        /// hpx::get_num_localities() participate and the local rank is
        /// hpx::get_locality_id().
        barrier(std::string base_name);

        /// Creates a barrier with a given size, rank is locality id
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        ///
        /// A barrier \param base_name is created. It expects that
        /// \param num participate and the local rank is
        /// hpx::get_locality_id().
        barrier(std::string base_name, std::size_t num);

        /// Creates a barrier with a given size and rank
        ///
        /// \param base_name The name of the barrier
        /// \param num The number of participating threads
        /// \param rank
        ///
        /// A barrier \param base_name is created. It expects that
        /// \param num participate and the local rank is \param rank.
        barrier(std::string base_name, std::size_t num, std::size_t rank);

        ~barrier();

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        void wait();

        /// Wait until each participant entered the barrier. Must be called by
        /// all participants
        ///
        /// \returns a future that becomes ready once all participants entered
        /// the barrier.
        hpx::future<void> wait_async();

        /// Resets this barrier instance.
        void release();

        /// Get the instance of the global barrier
        static barrier& get_global_barrier();

        /// Perform a global synchronization using the default global barrier
        /// The barrier is created once at startup and can be reused throughout
        /// the lifetime of an HPX application.
        ///
        /// \note This function currently does not support dynamic connection
        /// and disconnection of localities.
        static void synchronize();

    private:
        boost::intrusive_ptr<wrapping_type> node_;
    };
}}

#endif
