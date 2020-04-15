//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_JUN_27_2008_0820PM)
#define HPX_LCOS_JUN_27_2008_0820PM

#include <hpx/config.hpp>
#include <hpx/include/actions.hpp>

#include <hpx/lcos/base_lco.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>

#include <hpx/lcos/packaged_action.hpp>

#include <hpx/collectives/barrier.hpp>
#include <hpx/collectives/gather.hpp>
#include <hpx/collectives/latch.hpp>
#include <hpx/collectives/reduce.hpp>
#include <hpx/lcos/channel.hpp>

#include <hpx/async_combinators/split_future.hpp>
#include <hpx/async_combinators/wait_all.hpp>
#include <hpx/async_combinators/wait_any.hpp>
#include <hpx/async_combinators/wait_each.hpp>
#include <hpx/async_combinators/wait_some.hpp>
#include <hpx/async_combinators/when_all.hpp>
#include <hpx/async_combinators/when_any.hpp>
#include <hpx/async_combinators/when_each.hpp>
#include <hpx/async_combinators/when_some.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/dataflow.hpp>
#include <hpx/include/local_lcos.hpp>

#endif
