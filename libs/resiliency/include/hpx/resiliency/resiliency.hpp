//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/async_replay.hpp>
#include <hpx/resiliency/async_replay_executor.hpp>
#include <hpx/resiliency/async_replicate.hpp>
#include <hpx/resiliency/async_replicate_executor.hpp>
#include <hpx/resiliency/dataflow_replay.hpp>
#include <hpx/resiliency/dataflow_replicate.hpp>
#include <hpx/resiliency/version.hpp>

namespace hpx { namespace experimental {

    // Replay APIs
    using hpx::resiliency::abort_replay_exception;

    using hpx::resiliency::async_replay_validate;
    using hpx::resiliency::async_replay;

    using hpx::resiliency::dataflow_replay_validate;
    using hpx::resiliency::dataflow_replay;

    // Replicate APIs
    using hpx::resiliency::abort_replicate_exception;

    using hpx::resiliency::async_replicate_vote_validate;
    using hpx::resiliency::async_replicate_vote;
    using hpx::resiliency::async_replicate_validate;
    using hpx::resiliency::async_replicate;

    using hpx::resiliency::dataflow_replicate_vote_validate;
    using hpx::resiliency::dataflow_replicate_vote;
    using hpx::resiliency::dataflow_replicate_validate;
    using hpx::resiliency::dataflow_replicate;
}}    // namespace hpx::experimental
