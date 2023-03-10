//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)

// clang-format off
namespace hpx::collectives {

    /// The function \a create_communication_set sets up a (distributed)
    /// tree-like communication structure that can be used with any of the
    /// collective APIs (such like \a all_to_all and similar).
    ///
    /// \param  basename    The base name identifying the all_to_all operation
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
    /// \param arity        The number of children each of the communication
    ///                     nodes is connected to (default: picked based on
    ///                     num_sites).
    ///
    /// \returns    This function returns a new communicator object usable
    ///             with the collective operation.
    ///
    communicator create_communication_set(char const* basename,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        arity_arg arity = arity_arg());
}
// clang-format on

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/collectives/argument_types.hpp>
#include <hpx/collectives/create_communicator.hpp>

namespace hpx::collectives {

    struct communicator;

    HPX_EXPORT communicator create_communication_set(char const* name,
        num_sites_arg num_sites = num_sites_arg(),
        this_site_arg this_site = this_site_arg(),
        generation_arg generation = generation_arg(),
        arity_arg arity = arity_arg());
}    // namespace hpx::collectives

#endif    // DOXYGEN
