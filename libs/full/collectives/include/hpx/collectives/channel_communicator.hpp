//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file create_communicator.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(DOXYGEN)
// clang-format off
namespace hpx { namespace collectives {

    /// Create a new communicator object usable with peer-to-peer
    /// channel-based operations
    ///
    /// This functions creates a new communicator object that can be called in
    /// order to pre-allocate a communicator object usable with multiple
    /// invocations of channel-based peer-to-peer operations.
    ///
    /// \param basename     The base name identifying the collective operation
    /// \param num_sites    The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a future to a new communicator object
    ///             usable with the collective operation.
    ///
    hpx::future<channel_communicator> create_channel_communicator(
        char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg());

    /// Create a new communicator object usable with peer-to-peer
    /// channel-based operations
    ///
    /// This functions creates a new communicator object that can be called in
    /// order to pre-allocate a communicator object usable with multiple
    /// invocations of channel-based peer-to-peer operations.
    ///
    /// \param basename     The base name identifying the collective operation
    /// \param num_sites    The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    ///
    /// \returns    This function returns a new communicator object usable
    ///             with the collective operation.
    ///
    channel_communicator create_channel_communicator(
        hpx::launch::sync_policy, char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg());

    /// Send a value to the given site
    ///
    /// This function sends a value to the given site based on the given
    /// communicator.
    ///
    /// \param comm     The channel communicator object to use for the data
    ///                 transfer
    /// \param site     The destination site
    /// \param value    The value to send
    /// \param tag      The (optional) tag identifying the concrete operation
    ///
    /// \returns    This function returns a future<void> that becomes ready
    ///             once the data transfer operation has finished.
    ///
    template <typename T>
    hpx::future<void> set(channel_communicator comm,
        that_site_arg site, T&& value, tag_arg tag = tag_arg());

    /// Send a value to the given site
    ///
    /// This function receives a value from the given site based on the given
    /// communicator.
    ///
    /// \param comm     The channel communicator object to use for the data transfer
    /// \param site     The source site
    ///
    /// \returns    This function returns a future<T> that becomes ready once the
    ///             data transfer operation has finished. The future will hold the
    ///             received value.
    ///
    template <typename T>
    hpx::future<T> get(channel_communicator comm, that_site_arg site,
        tag_arg tag = tag_arg());

}}
// clang-format on

#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_distributed/async.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/detail/channel_communicator.hpp>
#include <hpx/components/client.hpp>
#include <hpx/futures/future.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace collectives {

    ///////////////////////////////////////////////////////////////////////////
    // forward declarations
    class channel_communicator;

    template <typename T>
    hpx::future<T> get(
        channel_communicator, that_site_arg, tag_arg = tag_arg());

    template <typename T>
    hpx::future<void> set(
        channel_communicator, that_site_arg, T&&, tag_arg = tag_arg());

    ///////////////////////////////////////////////////////////////////////////
    class channel_communicator
    {
    private:
        friend HPX_EXPORT hpx::future<channel_communicator>
        create_channel_communicator(char const* basename,
            num_sites_arg num_sites, this_site_arg this_site);

        template <typename T>
        friend hpx::future<T> get(channel_communicator, that_site_arg, tag_arg);

        template <typename T>
        friend hpx::future<void> set(
            channel_communicator, that_site_arg, T&&, tag_arg);

    private:
        HPX_EXPORT channel_communicator(char const* basename,
            num_sites_arg num_sites, this_site_arg this_site,
            components::client<detail::channel_communicator_server>&& here);

    public:
        HPX_EXPORT channel_communicator();

        channel_communicator(channel_communicator const& rhs) = default;
        channel_communicator& operator=(
            channel_communicator const& rhs) = default;

        channel_communicator(channel_communicator&& rhs) noexcept = default;
        channel_communicator& operator=(
            channel_communicator&& rhs) noexcept = default;

        HPX_EXPORT void free();

    private:
        std::shared_ptr<detail::channel_communicator> comm_;
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT hpx::future<channel_communicator> create_channel_communicator(
        char const* basename, num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg());

    HPX_EXPORT channel_communicator create_channel_communicator(
        hpx::launch::sync_policy, char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg());

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    hpx::future<T> get(
        channel_communicator comm, that_site_arg site, tag_arg tag)
    {
        return comm.comm_->template get<T>(site.argument_, tag.argument_);
    }

    template <typename T>
    hpx::future<void> set(
        channel_communicator comm, that_site_arg site, T&& value, tag_arg tag)
    {
        return comm.comm_->set(
            site.argument_, HPX_FORWARD(T, value), tag.argument_);
    }
}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
