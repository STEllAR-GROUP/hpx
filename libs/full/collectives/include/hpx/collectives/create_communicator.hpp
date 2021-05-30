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
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);
}}
// clang-format on

#else

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/components/client.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace collectives {

    using communicator = hpx::components::client<detail::communicator_server>;

    ///////////////////////////////////////////////////////////////////////////
    HPX_EXPORT communicator create_communicator(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t generation = std::size_t(-1),
        std::size_t this_site = std::size_t(-1), std::size_t root_site = 0);

}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
#endif    // DOXYGEN
