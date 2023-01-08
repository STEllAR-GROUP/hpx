//  Copyright (C) 2008, 2016 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  tagged pointer, for aba prevention

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_LOCKFREE_PTR_COMPRESSION)
#include <hpx/concurrency/detail/tagged_ptr_dcas.hpp>
#else
#include <hpx/concurrency/detail/tagged_ptr_ptrcompression.hpp>
#endif
