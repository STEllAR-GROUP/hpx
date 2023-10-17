//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file create_communicator.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(DOXYGEN)
// clang-format off
/// Top level HPX namespace
namespace hpx { namespace collectives {

    /// A communicator instance represents the list of sites that participate in
    /// a particular collective operation.
    struct communicator
    {
        /// Store the number of used sites and the index of the current site for
        /// this communicator instance.
        ///
        /// \param  num_sites   The number of participating sites (default: all
        ///                     localities).
        /// \param this_site    The sequence number of this site (usually
        ///                     the locality id).
        void set_info(
            num_sites_arg num_sites, this_site_arg this_site) noexcept;

        /// Retrieve the number of used sites and the index of the current site
        /// for this communicator instance.
        [[nodiscard]] std::pair<num_sites_arg, this_site_arg>
        get_info() const noexcept;

        /// Return whether this communicator instance represents the root site
        /// of the communication operation.
        [[nodiscard]] bool is_root() const;
    };

    /// Create a new communicator object usable with any collective operation
    ///
    /// This functions creates a new communicator object that can be called in
    /// order to pre-allocate a communicator object usable with multiple
    /// invocations of any of the collective operations (such as \a all_gather,
    /// \a all_reduce, \a all_to_all, \a broadcast, etc.).
    ///
    /// \param  basename    The base name identifying the collective operation
    /// \param  num_sites   The number of participating sites (default: all
    ///                     localities).
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the collective operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the collective operation on the
    ///                     given base name has to be performed more than once.
    /// \param  root_site   The site that is responsible for creating the
    ///                     collective support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a new communicator object usable
    ///             with the collective operation.
    ///
    communicator create_communicator(char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    /// Create a new communicator object usable with any local collective
    /// operation
    ///
    /// This functions creates a new communicator object that can be called in
    /// order to pre-allocate a communicator object usable with multiple
    /// invocations of any of the collective operations (such as \a all_gather,
    /// \a all_reduce, \a all_to_all, \a broadcast, etc.).
    ///
    /// \param  basename    The base name identifying the collective operation
    /// \param  num_sites   The number of participating sites
    /// \param  this_site   The sequence number of this invocation (usually
    ///                     the sequence number of the object participating in
    ///                     the collective operation). This value must be in the
    ///                     range [0, num_sites).
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the collective operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the collective operation on the
    ///                     given base name has to be performed more than once.
    /// \param  root_site   The site that is responsible for creating the
    ///                     collective support object. This value is optional
    ///                     and defaults to '0' (zero).
    ///
    /// \returns    This function returns a new communicator object usable
    ///             for all local collective operations.
    ///
    communicator create_local_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());
}}
// clang-format on

#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/components/client_base.hpp>
#include <hpx/type_support/extra_data.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::collectives::detail {

    // Data stored in the shared state of the communicator client type below
    struct communicator_data
    {
        num_sites_arg num_sites_;
        this_site_arg this_site_;
    };
}    // namespace hpx::collectives::detail

// This is explicitly instantiated to ensure that the id is stable across
// shared libraries.
template <>
struct hpx::util::extra_data_helper<hpx::collectives::detail::communicator_data>
{
    HPX_EXPORT static extra_data_id_type id() noexcept;
    static constexpr void reset(
        collectives::detail::communicator_data*) noexcept
    {
    }
};    // namespace hpx::util

namespace hpx::collectives {

    struct communicator
      : hpx::components::client_base<communicator, detail::communicator_server,
            detail::communicator_data>
    {
        using base_type = client_base<communicator, detail::communicator_server,
            detail::communicator_data>;
        using future_type = base_type::future_type;

    public:
        // construction
        communicator() = default;

        explicit communicator(base_type const& base) noexcept
          : base_type(base)
        {
        }
        explicit communicator(base_type&& base) noexcept
          : base_type(HPX_MOVE(base))
        {
        }
        explicit communicator(hpx::id_type&& id)
          : base_type(HPX_MOVE(id))
        {
        }
        explicit communicator(future<hpx::id_type>&& id) noexcept
          : base_type(HPX_MOVE(id))
        {
        }
        communicator(future<communicator>&& c)
          : base_type(HPX_MOVE(c))
        {
        }

        HPX_EXPORT void set_info(
            num_sites_arg num_sites, this_site_arg this_site) noexcept;

        [[nodiscard]] HPX_EXPORT std::pair<num_sites_arg, this_site_arg>
        get_info() const noexcept;

        [[nodiscard]] bool is_root() const
        {
            return !base_type::registered_name().empty();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT communicator create_communicator(char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

    HPX_EXPORT communicator create_local_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
