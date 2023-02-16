//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>

namespace hpx::parallel::util {

    inline constexpr std::uint32_t tmsb[256] = {0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4,
        4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8};

    // \brief Obtain the number of bits equal or greater than num
    // \param [in] num : Number to examine
    // \exception none
    // \return Number of bits
    constexpr std::uint32_t nbits32(std::uint32_t num) noexcept
    {
        int Pos = num & 0xffff0000U ? 16 : 0;
        if (num >> Pos & 0xff00U)
        {
            Pos += 8;
        }
        return tmsb[num >> Pos] + Pos;
    }

    // \brief Obtain the number of bits equal or greater than num
    // \param [in] num : Number to examine
    // \exception none
    // \return Number of bits
    constexpr std::uint32_t nbits64(std::uint64_t num) noexcept
    {
        uint32_t pos = num & 0xffffffff00000000ULL ? 32 : 0;
        if (num >> pos & 0xffff0000ULL)
        {
            pos += 16;
        }
        if (num >> pos & 0xff00ULL)
        {
            pos += 8;
        }
        return tmsb[num >> pos] + pos;
    }
}    // namespace hpx::parallel::util
