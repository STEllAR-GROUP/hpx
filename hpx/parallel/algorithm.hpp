//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file algorithm.hpp

#if !defined(HPX_PARALLEL_ALGORITHM_MAY_28_2014_0522PM)
#define HPX_PARALLEL_ALGORITHM_MAY_28_2014_0522PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/detail/dispatch.hpp>

/// See N4071: 1.3/3
#include <algorithm>

#include <hpx/parallel/detail/all_any_none.hpp>
#include <hpx/parallel/detail/copy.hpp>
#include <hpx/parallel/detail/count.hpp>
#include <hpx/parallel/detail/equal.hpp>
#include <hpx/parallel/detail/fill.hpp>
#include <hpx/parallel/detail/for_each.hpp>
#include <hpx/parallel/detail/move.hpp>
#include <hpx/parallel/detail/reduce.hpp>
#include <hpx/parallel/detail/swap_ranges.hpp>
#include <hpx/parallel/detail/transform.hpp>

#undef HPX_PARALLEL_DISPATCH

#endif

