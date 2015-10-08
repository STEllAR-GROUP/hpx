//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file agas_fwd.hpp

#ifndef HPX_RUNTIME_AGAS_FWD_HPP
#define HPX_RUNTIME_AGAS_FWD_HPP

#include <hpx/config/export_definitions.hpp>

namespace hpx {
    namespace agas
    {
        struct HPX_API_EXPORT addressing_service;

        enum service_mode
        {
            service_mode_invalid = -1,
            service_mode_bootstrap = 0,
            service_mode_hosted = 1
        };
    }
}

#endif
