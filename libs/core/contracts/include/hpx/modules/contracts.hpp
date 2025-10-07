//  Copyright (c) 2025 The STE||AR-Group
//  Copyright (c) 2025 Alexandros Papadakis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file
/// \page HPX_PRE, HPX_POST, HPX_CONTRACT_ASSERT
/// \headerfile hpx/contracts.hpp
///
/// This header provides C++ contracts support for HPX with intelligent fallback
/// to assertions when contracts are not available.
///
/// ## API Reference:
/// - **HPX_PRE(condition)**: Precondition contracts
/// - **HPX_POST(condition)**: Postcondition contracts  
/// - **HPX_CONTRACT_ASSERT(condition)**: Contract assertions (always available)
///
/// ## Configuration:
/// Enable with: `cmake -DHPX_WITH_CONTRACTS=ON -DCMAKE_CXX_STANDARD=26`
///
/// See docs/index.rst for comprehensive usage guide.

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>

// Contract implementation: automatically selects native C++26 contracts 
// or falls back to HPX_ASSERT based on compiler capabilities
#ifdef HPX_HAVE_CONTRACTS
    #if HPX_HAVE_NATIVE_CONTRACTS
        // Native C++26 contracts mode
        #define HPX_PRE(x) pre(x)
        #define HPX_CONTRACT_ASSERT(x) contract_assert(x)
        #define HPX_POST(x) post(x)
    #else
        // Fallback mode: contracts map to assertions until C++26 migration
        #pragma message("HPX Contracts: Using assertion fallback mode. " \
                       "Contracts will map to HPX_ASSERT until C++26 native support is available.")
        #define HPX_PRE(x) HPX_ASSERT((x))
        #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))  
        #define HPX_POST(x) HPX_ASSERT((x)) 
    #endif
#else
    // Contracts disabled: PRE/POST are no-ops, CONTRACT_ASSERT remains available
    #define HPX_PRE(x)
    #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))
    #define HPX_POST(x)
#endif

#ifdef HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS 
    // Override HPX_ASSERT to use contract assertions 
    #define HPX_ASSERT(x) HPX_CONTRACT_ASSERT(x)
#endif
    