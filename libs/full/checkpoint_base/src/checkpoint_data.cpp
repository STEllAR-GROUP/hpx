// Copyright (c) 2018 Adrian Serio
// Copyright (c) 2018-2020 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/checkpoint_base/checkpoint_data.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>

namespace hpx { namespace serialization { namespace detail {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    void
    extra_archive_data_id_helper<hpx::util::checkpointing_tag>::id() noexcept
    {
    }
}}}    // namespace hpx::serialization::detail
