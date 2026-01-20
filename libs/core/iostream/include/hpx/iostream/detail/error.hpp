//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>

#include <ios>

namespace hpx::iostream::detail {

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure cant_read()
    {
        return std::ios_base::failure("no read access");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure cant_write()
    {
        return std::ios_base::failure("no write access");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure cant_seek()
    {
        return std::ios_base::failure("no random access");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure bad_read()
    {
        return std::ios_base::failure("bad read");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure bad_putback()
    {
        return std::ios_base::failure("putback buffer full");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure bad_write()
    {
        return std::ios_base::failure("bad write");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure write_area_exhausted()
    {
        return std::ios_base::failure("write area exhausted");
    }

    HPX_CXX_CORE_EXPORT inline std::ios_base::failure bad_seek()
    {
        return std::ios_base::failure("bad seek");
    }
}    // namespace hpx::iostream::detail
