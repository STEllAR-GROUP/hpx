//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/datastructures/detail/dynamic_bitset.hpp>
#include <hpx/serialization.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx::detail {

    template <typename Block, typename Alloc>
    void serialize(hpx::serialization::output_archive& ar,
        dynamic_bitset<Block, Alloc> const& bs, unsigned)
    {
        ar << static_cast<std::uint64_t>(bs.nubits_);
        ar << bs.bits_;
    }

    template <typename Block, typename Alloc>
    void serialize(hpx::serialization::input_archive& ar,
        dynamic_bitset<Block, Alloc>& bs, unsigned)
    {
        std::uint64_t nubits;
        ar >> nubits;
        bs.nubits_ = static_cast<std::size_t>(nubits);
        ar >> bs.bits_;
    }
}    // namespace hpx::detail
