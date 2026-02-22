//  Copyright (c) 2025 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Smoke-test: verify that HPX can be used via C++ Modules (distributed runtime).
// This file is only compiled when HPX_WITH_CXX_MODULES=ON and
// HPX_WITH_DISTRIBUTED_RUNTIME=ON.

// To maintain ABI compatibility and expose necessary macros defined by HPX,
// users still need to #include HPX headers in addition to the BMI.
#include <hpx/hpx.hpp>

int hpx_main()
{
    // Minimal test: initialise and finalize the HPX distributed runtime via
    // module import.
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
