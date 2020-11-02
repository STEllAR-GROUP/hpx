//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/dataflow.hpp>
#include <hpx/include/lcos_local.hpp>
#include <hpx/modules/async_combinators.hpp>

#if defined(HPX_HAVE_DISTRIBUTED_RUNTIME)
#include <hpx/collectives/barrier.hpp>
#include <hpx/collectives/gather.hpp>
#include <hpx/collectives/latch.hpp>
#include <hpx/collectives/reduce.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/lcos/packaged_action.hpp>
#include <hpx/lcos_distributed/channel.hpp>
#endif
