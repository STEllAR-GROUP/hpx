////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2024 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/topology/cpu_mask.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx::threads {

    HPX_CORE_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, std::size_t used_cores,
        std::size_t max_cores, std::size_t num_threads,
        std::vector<std::size_t>& num_pus, bool use_process_mask,
        error_code& ec = throws);

    // backwards compatibility helper
    inline void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec = throws)
    {
        std::vector<std::size_t> num_pus;
        parse_affinity_options(
            spec, affinities, 1, 1, affinities.size(), num_pus, false, ec);
    }
}    // namespace hpx::threads
