//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

/// \cond NODETAIL
namespace hpx { namespace cds {

    ///////////////////////////////////////////////////////////////////////////
    // this wrapper will initialize a std::thread for use with libCDS
    // algorithms
    struct HPX_PARALLELISM_EXPORT stdthread_manager_wrapper
    {
        explicit stdthread_manager_wrapper();
        ~stdthread_manager_wrapper();
    };
}}    // namespace hpx::cds
