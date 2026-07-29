//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/components.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/naming_base.hpp>

#include <hpx/supervision_dispatch/server/sentinel.hpp>

#include <cstdint>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::supervision {

    ///////////////////////////////////////////////////////////////////////////
    // A sentinel is a lightweight, self-supervising handle: constructing it
    // creates a component on the given target locality, and calling start()
    // publishes the `started` lifecycle event for that component to the
    // supervision manager running on the same locality. No registry lookup or
    // discovery step is required.
    class HPX_SUPERVISION_DISPATCH_EXPORT sentinel
      : public hpx::components::client_base<sentinel, server::sentinel>
    {
        using base_type =
            hpx::components::client_base<sentinel, server::sentinel>;

    public:
        explicit sentinel(
            hpx::id_type const& target_locality = hpx::invalid_id);
        /* implicit */ sentinel(hpx::future<hpx::id_type>&& f);

        // Publish the `started` lifecycle event (at the given epoch) for
        // this sentinel to the supervision manager running on its target
        // locality.
        hpx::future<publish_result> start(std::uint64_t epoch = 0) const;
        publish_result start(hpx::launch::sync_policy, std::uint64_t epoch = 0,
            hpx::error_code& ec = hpx::throws) const;

/// Register this sentinel's id with AGAS under a name pinned to the
        /// locality that actually hosts its underlying server component
        /// (derived from the component's own id via
        /// hpx::naming::get_locality_id_from_id(), not the ambient
        /// hpx::get_locality_id() of the caller), so that discovery of this
        /// sentinel is unaffected by failures on any locality other than the
        /// one it actually lives on. May only be called once per sentinel
        /// instance, mirroring the call-once contract of the underlying
        /// client_base::register_as().
        ///
        /// \return A future that becomes ready with `true` if the name was
        ///         registered successfully, `false` otherwise.
        hpx::future<bool> register_basename();

        /// \copydoc register_basename()
        ///
        /// \param ec Used to hold the error code that results from the
        ///           operation instead of throwing an exception on failure.
        /// \return `true` if the name was registered successfully, `false`
        ///         otherwise.
        bool register_basename(
            hpx::launch::sync_policy, hpx::error_code& ec = hpx::throws);

        /// Remove this sentinel's AGAS name registration again, using the name
        /// most recently established by register_basename() (retrieved via
        /// registered_name(), the single source of truth client_base already
        /// maintains for it). Calling this explicitly is safe even though
        /// destroying this sentinel also unregisters its name automatically
        /// (client_base::register_as()'s default manage_lifetime=true): the
        /// redundant unregister attempt on destruction is a silent no-op.
        ///
        /// \return A future that becomes ready with the id that was registered
        ///         under the removed name.
        hpx::future<hpx::id_type> unregister_basename() const;

        /// \copydoc unregister_basename()
        ///
        /// \param ec Used to hold the error code that results from the
        ///           operation instead of throwing an exception on failure.
        /// \return The id that was registered under the removed name.
        hpx::id_type unregister_basename(
            hpx::launch::sync_policy, hpx::error_code& ec = hpx::throws) const;    };
}    // namespace hpx::supervision

#include <hpx/config/warnings_suffix.hpp>
