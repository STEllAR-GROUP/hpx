//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file unmanaged.hpp

#if !defined(HPX_NAMING_UNMANAGED_NOV_12_2013_0210PM)
#define HPX_NAMING_UNMANAGED_NOV_12_2013_0210PM

#include <hpx/runtime/naming/name.hpp>

namespace hpx { namespace naming
{
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
    inline id_type unmanaged(id_type const& id)
    {
        return id_type(detail::strip_internal_bits_from_gid(id.get_msb()),
            id.get_lsb(), id_type::unmanaged);
    }
}}

namespace hpx
{
    using naming::unmanaged;
}

#endif


