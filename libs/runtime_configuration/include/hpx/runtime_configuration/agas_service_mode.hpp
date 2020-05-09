//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace hpx { namespace agas {
    enum service_mode
    {
        service_mode_invalid = -1,
        service_mode_bootstrap = 0,
        service_mode_hosted = 1
    };
}}    // namespace hpx::agas
