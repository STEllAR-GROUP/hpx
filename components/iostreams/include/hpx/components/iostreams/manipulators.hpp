//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2013-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/iostreams/export_definitions.hpp>

#include <ostream>

namespace hpx::iostreams {

    struct flush_type
    {
    };    // hpx::flush
    struct endl_type
    {
    };    // hpx::endl
    struct async_flush_type
    {
    };    // hpx::async_flush
    struct async_endl_type
    {
    };    // hpx::async_endl

    inline std::ostream& operator<<(std::ostream& os, flush_type const&)
    {
        return os << std::flush;
    }

    inline std::ostream& operator<<(std::ostream& os, endl_type const&)
    {
        return os << std::endl;
    }

    inline std::ostream& operator<<(std::ostream& os, async_flush_type const&)
    {
        return os << std::flush;
    }

    inline std::ostream& operator<<(std::ostream& os, async_endl_type const&)
    {
        return os << std::endl;
    }
}    // namespace hpx::iostreams
