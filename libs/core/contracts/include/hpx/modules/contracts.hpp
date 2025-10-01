//  Copyright (c) 2025 The STE||AR-Group
//  Copyright (c) 2025 Alexandros Papadakis
//  Copyright (c) 2025 Panagiotis Syskakis
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
/// ## Design Goals:
/// - Use native C++26 contracts when available and enabled
/// - Graceful fallback to HPX_ASSERT for compatibility  
/// - Zero overhead when contracts are disabled
/// - Consistent API across all configurations
///
/// ## Configuration Matrix:
/// 
/// | HPX_HAVE_CONTRACTS | C++ Version | Behavior |
/// |--------------------|-------------|----------|
/// | ON                 | C++26+      | Native contracts |
/// | ON                 | < C++26     | HPX_ASSERT fallback |
/// | OFF                | Any         | No-op (except HPX_CONTRACT_ASSERT) |
///
/// ## Important Migration Notes:
///
/// **⚠️ Current Implementation Status:**
/// This implementation is currently in transition mode. The existing test code
/// uses contracts within function bodies, but for full C++26 compatibility:
///
/// - **HPX_PRE()** and **HPX_POST()** must be moved from function bodies to 
///   function declarations when using C++26 native contracts
/// - **HPX_POST()** in fallback mode cannot access return values (no 'r' parameter)
/// - Current test files use in-body syntax which works in fallback mode but
///   will need migration for native C++26 contract syntax
///
/// **Migration Path:**
/// 1. Current: `HPX_PRE(condition)` inside function body (fallback mode)
/// 2. Target: `int func() HPX_PRE(condition)` in declaration (C++26 mode)
///
/// ## Usage Examples:
///
/// ### Preconditions:
/// \code
/// // C++26 mode: Use in function declaration
/// int divide(int a, int b) HPX_PRE(b != 0)
/// {
///     return a / b;
/// }
///
/// // Fallback mode: Use in function body
/// int divide(int a, int b)
/// {
///     HPX_PRE(b != 0);
///     return a / b;
/// }
/// \endcode
///
/// ### Postconditions:
/// \code
/// // C++26 mode: Can access return value as 'r'
/// int factorial(int n) HPX_PRE(n >= 0) HPX_POST(r; r > 0)
/// {
///     return n <= 1 ? 1 : n * factorial(n - 1);
/// }
///
/// // Current fallback mode: Cannot access return value
/// int factorial(int n)
/// {
///     HPX_PRE(n >= 0);
///     int result = n <= 1 ? 1 : n * factorial(n - 1);
///     // HPX_POST(result > 0);  // Must use local variable, not 'r'
///     return result;
/// }
/// \endcode
///
/// **Note:** Current implementation has postconditions in function body without
/// return value access. Full C++26 migration will enable proper postcondition
/// syntax with return value checking.
///
/// ### Contract Assertions:
/// \code
/// void process_array(std::vector<int>& arr, size_t index)
/// {
///     HPX_CONTRACT_ASSERT(index < arr.size());
///     arr[index] *= 2;
/// }
/// \endcode
///
/// ## Build Configuration:
/// 
/// Enable contracts in CMake:
/// \code
/// cmake -DHPX_WITH_CONTRACTS=ON -DCMAKE_CXX_STANDARD=26
/// \endcode
///
/// Optional: Unify assertion systems:
/// \code  
/// cmake -DHPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS=ON
/// \endcode

#pragma once

#include <hpx/config.hpp>

// Include HPX_ASSERT only when we need it for fallback mode
// This avoids unnecessary dependencies when using native C++26 contracts
#if !defined(HPX_HAVE_CONTRACTS) || __cplusplus < 202602L
#include <hpx/assert.hpp>
#endif

#ifdef HPX_HAVE_CONTRACTS
    #if __cplusplus >= 202602L
        #define HPX_PRE(x) pre((x))
        #define HPX_CONTRACT_ASSERT(x) contract_assert((x))
        #define HPX_POST(x) post((x))


        #ifdef HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS 
            #define HPX_ASSERT(x) HPX_CONTRACT_ASSERT((x))
        #endif


    #else
        #pragma message("Warning: Contracts require C++26 or later. Falling back to HPX_ASSERT.")        
        #define HPX_PRE(x) HPX_ASSERT((x))
        #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))  
        #define HPX_POST(x) HPX_ASSERT((x)) 
    #endif

#else
    #define HPX_PRE(x)
    #if defined(HPX_ASSERT) 
        #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))
    #else 
        #define HPX_CONTRACT_ASSERT(x)
    #endif
    #define HPX_POST(x)
#endif


