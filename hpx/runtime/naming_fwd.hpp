//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_NAMING_FWD_HPP
#define HPX_RUNTIME_NAMING_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>

#include <cstdint>

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
        struct HPX_EXPORT id_type;
        struct HPX_API_EXPORT address;

        HPX_API_EXPORT resolver_client& get_agas_client();

        typedef std::uint64_t address_type;

        HPX_CONSTEXPR_OR_CONST std::uint32_t invalid_locality_id = ~0U;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Pulling important types into the main namespace
    using naming::id_type;
}

#endif
