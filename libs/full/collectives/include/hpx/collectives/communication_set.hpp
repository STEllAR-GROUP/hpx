//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)

// clang-format off
namespace hpx { namespace lcos {

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
    /// \param arity        The number of children each of the communication
    ///                     nodes is connected to (default: picked based on
    ///                     num_sites)
    ///
    /// \returns    This function returns a future holding an id_type of the
    ///             communicator object to be used on the current locality.
    ///
    hpx::future<hpx::id_type> create_communication_set(char const* basename,
        std::size_t num_sites = std::size_t(-1),
        std::size_t this_site = std::size_t(-1),
        std::size_t arity = std::size_t(-1));
}}
// clang-format on

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstddef>

namespace hpx { namespace lcos {

    HPX_EXPORT hpx::future<hpx::id_type> create_communication_set(
        char const* name, std::size_t num_sites = std::size_t(-1),
        std::size_t this_site = std::size_t(-1),
        std::size_t arity = std::size_t(-1));

}}    // namespace hpx::lcos

#endif    // DOXYGEN
