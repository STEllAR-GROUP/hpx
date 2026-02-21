//  Copyright (c) 2025 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Smoke-test: verify that HPX can be used via C++ Modules (local runtime only).
// This file is only compiled when HPX_WITH_CXX_MODULES=ON.

import HPX.Core;

// hpx_init.hpp cannot be part of the module (it contains macros for main()
// wrapping), so it is included textually.
#include <hpx/hpx_init.hpp>

int hpx_main()
{
    // Minimal test: initialise and finalize the HPX runtime via module import.
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
