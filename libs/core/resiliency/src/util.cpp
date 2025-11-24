//  Copyright (c) 2020 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/resiliency/util.hpp>

namespace hpx::resiliency::experimental {

    ///////////////////////////////////////////////////////////////////////////
    abort_replicate_exception::abort_replicate_exception() = default;
    abort_replicate_exception::~abort_replicate_exception() = default;

    abort_replicate_exception::abort_replicate_exception(
        abort_replicate_exception const&) = default;
    abort_replicate_exception& abort_replicate_exception::operator=(
        abort_replicate_exception const&) = default;

    ///////////////////////////////////////////////////////////////////////////
    abort_replay_exception::abort_replay_exception() = default;
    abort_replay_exception::~abort_replay_exception() = default;

    abort_replay_exception::abort_replay_exception(
        abort_replay_exception const&) = default;
    abort_replay_exception& abort_replay_exception::operator=(
        abort_replay_exception const&) = default;
}    // namespace hpx::resiliency::experimental
