//  Copyright (c) 2020-2021 Hartmut Kaiser
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
    /// \param  generation  The generational counter identifying the sequence
    ///                     number of the collective operation performed on the
    ///                     given base name. This is optional and needs to be
    ///                     supplied only if the collective operation on the
    ///                     given base name has to be performed more than once.
    /// \param this_site    The sequence number of this invocation (usually
    ///                     the locality id). This value is optional and
    ///                     defaults to whatever hpx::get_locality_id() returns.
    /// \params root_site   The site that is responsible for creating the
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
}}
// clang-format on

#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/components/client_base.hpp>

#include <cstddef>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace collectives {

    ///////////////////////////////////////////////////////////////////////////
    struct communicator
      : hpx::components::client_base<communicator, detail::communicator_server>
    {
        using base_type =
            client_base<communicator, detail::communicator_server>;
        using future_type = typename base_type::future_type;

    public:
        // construction
        communicator() = default;

        explicit communicator(naming::id_type&& id)
          : base_type(std::move(id))
        {
        }
        explicit communicator(future<hpx::id_type>&& id) noexcept
          : base_type(std::move(id))
        {
        }
        communicator(future<communicator>&& c)
          : base_type(std::move(c))
        {
        }

        void set_info(std::size_t num_sites, std::size_t this_site) noexcept
        {
            num_sites_ = num_sites;
            this_site_ = this_site;
        }

        std::pair<std::size_t, std::size_t> get_info() const noexcept
        {
            return std::make_pair(num_sites_, this_site_);
        }

        bool is_root() const
        {
            return !base_type::registered_name().empty();
        }

    private:
        std::size_t num_sites_ = std::size_t(-1);
        std::size_t this_site_ = std::size_t(-1);
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT communicator create_communicator(char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        root_site_arg root_site = root_site_arg());

}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
