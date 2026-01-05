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

namespace hpx::iostreams {

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
        return {std::mbstate_t(), off};
    }

    //------------------Definition of position_to_offset--------------------------//

    // Hande custom pos_type's
    HPX_CXX_CORE_EXPORT template <typename PosType>
    stream_offset position_to_offset(PosType pos)
    {
        return static_cast<stream_offset>(pos);
    }

    // Converts a std::fpos_t to a stream_offset
    HPX_CXX_CORE_EXPORT inline stream_offset fpos_t_to_offset(std::fpos_t pos)
    {
#if defined(_POSIX_) || (_INTEGRAL_MAX_BITS >= 64) || defined(__IBMCPP__)
        return pos;
#else
#if !defined(_FPOSOFF)
        return (long long) (pos);
#else
        return _FPOSOFF(pos);
#endif
#endif
    }

    // Extracts the member _Fpos from a std::fpos
    HPX_CXX_CORE_EXPORT inline std::fpos_t streampos_to_fpos_t(
        std::streampos pos)
    {
#if defined(_CPPLIB_VER) || defined(__IBMCPP__)
        return pos;
#else
        return pos.get_fpos_t();
#endif
    }

    HPX_CXX_CORE_EXPORT inline stream_offset position_to_offset(
        std::streampos const& pos)
    {
#if !defined(_FPOSOFF)
        return fpos_t_to_offset(streampos_to_fpos_t(pos)) +
            static_cast<stream_offset>(static_cast<std::streamoff>(pos) -
                (long long) (streampos_to_fpos_t(pos)));
#else
        return fpos_t_to_offset(streampos_to_fpos_t(pos)) +
            static_cast<stream_offset>(static_cast<std::streamoff>(pos) -
                _FPOSOFF(streampos_to_fpos_t(pos)));
#endif
    }
}    // namespace hpx::iostreams

#include <hpx/config/warnings_suffix.hpp>
