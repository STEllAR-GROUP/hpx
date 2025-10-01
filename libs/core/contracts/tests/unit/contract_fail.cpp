//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/contracts.hpp>

// This test only runs in fallback mode (HPX_WITH_CONTRACTS=ON && __cplusplus < 202602L)
// In fallback mode: HPX_CONTRACT_ASSERT(false) -> HPX_ASSERT(false) -> Should abort in Debug
// Expected to fail in Debug builds (WILL_FAIL property set in CMakeLists.txt)

int main()
{
    HPX_CONTRACT_ASSERT(false);
}