//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/serialization_fwd.hpp>

#if defined(HPX_SERIALIZATION_HAVE_BOOST_TYPES)
#include <hpx/serialization/vector.hpp>

#include <cstddef>
#include <vector>

#include <boost/dynamic_bitset.hpp>

namespace hpx { namespace serialization {

    template <typename Block, typename Alloc>
    void serialize(output_archive& ar,
        boost::dynamic_bitset<Block, Alloc> const& bs, unsigned)
    {
        std::size_t num_bits = bs.size();
        std::vector<Block> blocks(bs.num_blocks());
        boost::to_block_range(bs, blocks.begin());

        ar << num_bits;
        ar << blocks;
    }

    template <typename Block, typename Alloc>
    void serialize(
        input_archive& ar, boost::dynamic_bitset<Block, Alloc>& bs, unsigned)
    {
        std::size_t num_bits;
        std::vector<Block> blocks;
        ar >> num_bits;
        ar >> blocks;

        bs.resize(num_bits);
        boost::from_block_range(blocks.begin(), blocks.end(), bs);
        bs.resize(num_bits);
    }
}}    // namespace hpx::serialization

#endif
