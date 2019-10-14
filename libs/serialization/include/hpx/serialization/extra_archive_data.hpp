//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_EXTRA_ARCHIVE_DATA_HPP)
#define HPX_SERIALIZATION_EXTRA_ARCHIVE_DATA_HPP

#include <hpx/datastructures.hpp>

#include <vector>

namespace hpx { namespace serialization {

    using extra_archive_data_type = std::vector<util::unique_any_nonser>;
}}    // namespace hpx::serialization

#endif
