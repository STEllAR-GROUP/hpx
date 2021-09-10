//  Copyright (c) 2015-2021 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/naming/detail/preprocess_gid_types.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>

#include <cstdint>

namespace hpx { namespace serialization { namespace detail {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    extra_archive_data_id_type
    extra_archive_data_helper<preprocess_gid_types>::id() noexcept
    {
        static std::uint8_t id;
        return &id;
    }
}}}    // namespace hpx::serialization::detail
