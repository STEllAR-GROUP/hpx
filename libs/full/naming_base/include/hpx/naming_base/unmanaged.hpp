//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file unmanaged.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/naming_base/id_type.hpp>

namespace hpx { namespace naming {

    /// The helper function \a hpx::unmanaged can be used to generate
    /// a global identifier which does not participate in the automatic
    /// garbage collection.
    ///
    /// \param id   [in] The id to generated the unmanaged global id from
    ///             This parameter can be itself a managed or a unmanaged
    ///             global id.
    ///
    /// \returns    This function returns a new global id referencing the
    ///             same object as the parameter \a id. The only difference
    ///             is that the returned global identifier does not participate
    ///             in the automatic garbage collection.
    ///
    /// \note       This function allows to apply certain optimizations to
    ///             the process of memory management in HPX. It however requires
    ///             the user to take full responsibility for keeping the referenced
    ///             objects alive long enough.
    ///
    HPX_EXPORT id_type unmanaged(id_type const& id);
}}    // namespace hpx::naming

namespace hpx {
    using naming::unmanaged;
}
