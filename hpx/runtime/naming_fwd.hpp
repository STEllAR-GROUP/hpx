//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file naming_fwd.hpp

#ifndef HPX_RUNTIME_NAMING_FWD_HPP
#define HPX_RUNTIME_NAMING_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>

#include <boost/cstdint.hpp>

namespace hpx
{
    /// \namespace naming
    ///
    /// The namespace \a naming contains all definitions needed for the AGAS
    /// (Active Global Address Space) service.
    namespace naming
    {
        typedef agas::addressing_service resolver_client;

        struct HPX_EXPORT gid_type;
        // NOTE: We do not export the symbol here as id_type was already
        //       exported and generates a warning on gcc otherwise.
        struct id_type;
        struct HPX_API_EXPORT address;

        HPX_API_EXPORT resolver_client& get_agas_client();

        typedef boost::uint64_t address_type;

        HPX_CONSTEXPR_OR_CONST boost::uint32_t invalid_locality_id = ~0U;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Pulling important types into the main namespace
    using naming::id_type;
}

#endif
