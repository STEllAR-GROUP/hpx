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

#include <codecvt>
#include <cstdint>
#include <ios>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostream {

    //------------------Definition of stream_offset-------------------------------//
    HPX_CXX_CORE_EXPORT using stream_offset = std::intmax_t;

    //------------------Definition of stream_offset_to_streamoff------------------//
    HPX_CXX_CORE_EXPORT inline std::streamoff stream_offset_to_streamoff(
        stream_offset off)
    {
        return static_cast<std::streamoff>(off);
    }

    //------------------Definition of offset_to_position--------------------------//
    HPX_CXX_CORE_EXPORT inline std::streampos offset_to_position(
        stream_offset off)
    {
        return {off};
    }

    //------------------Definition of position_to_offset--------------------------//

    // Hande custom pos_type's
    HPX_CXX_CORE_EXPORT template <typename PosType>
    stream_offset position_to_offset(PosType pos)
    {
        return static_cast<stream_offset>(pos);
    }
}    // namespace hpx::iostream

#include <hpx/config/warnings_suffix.hpp>
