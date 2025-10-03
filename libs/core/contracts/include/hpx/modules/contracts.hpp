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
/// | HPX_HAVE_CONTRACTS | __cpp_contracts | Behavior |
/// |--------------------|-----------------|----------|
/// | ON                 | Available       | Native C++26 contracts |
/// | ON                 | Not Available   | HPX_ASSERT fallback |
/// | OFF                | Any             | HPX_CONTRACT_ASSERT → HPX_ASSERT, PRE/POST → no-op |
///
/// ## Design Philosophy:
///
/// **HPX_CONTRACT_ASSERT** = Enhanced assertion that's always useful when assertions work
/// - Available even when HPX_WITH_CONTRACTS=OFF because it's fundamentally a better assertion
/// - Maps to HPX_ASSERT, so works whenever assertions are active (Debug builds)
///
/// **HPX_PRE/HPX_POST** = True contract syntax that requires language support
/// - Disabled when HPX_WITH_CONTRACTS=OFF because they won't work the same in C++26
/// - In C++26: attached to function declarations, not usable in function bodies
/// - In fallback: work in function bodies but this is temporary compatibility
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
/// // C++26 native contracts: Use in function declaration
/// int divide(int a, int b) HPX_PRE(b != 0)
/// {
///     return a / b;
/// }
///
/// // Fallback mode: Use in function body (temporary compatibility)
/// int divide(int a, int b)
/// {
///     HPX_PRE(b != 0);  // Maps to HPX_ASSERT(b != 0)
///     return a / b;
/// }
///
/// // Contracts disabled: PRE/POST are no-ops to prepare for C++26 migration
/// int divide(int a, int b)
/// {
///     HPX_PRE(b != 0);  // No-op - use HPX_CONTRACT_ASSERT instead
///     return a / b;
/// }
/// \endcode
///
/// ### Postconditions:
/// \code
/// // C++26 native: Can access return value as 'r'
/// int factorial(int n) HPX_PRE(n >= 0) HPX_POST(r; r > 0)
/// {
///     return n <= 1 ? 1 : n * factorial(n - 1);
/// }
///
/// // Fallback mode: Cannot access return value (temporary)
/// int factorial(int n)
/// {
///     HPX_PRE(n >= 0);
///     int result = n <= 1 ? 1 : n * factorial(n - 1);
///     HPX_POST(result > 0);  // Maps to HPX_ASSERT, use local variable
///     return result;
/// }
/// \endcode
///
/// ### Contract Assertions (Always Available):
/// \code
/// void process_array(std::vector<int>& arr, size_t index)
/// {
///     // HPX_CONTRACT_ASSERT works in all modes:
///     // - Native C++26: maps to contract_assert
///     // - Fallback: maps to HPX_ASSERT  
///     // - Contracts disabled: still maps to HPX_ASSERT (enhanced assertion)
///     HPX_CONTRACT_ASSERT(index < arr.size());
///     arr[index] *= 2;
/// }
///
/// // Recommended pattern when contracts are disabled:
/// void safe_function(int* ptr)
/// {
///     HPX_CONTRACT_ASSERT(ptr != nullptr);  // Always works
///     // HPX_PRE(ptr != nullptr);           // No-op when contracts disabled
///     *ptr = 42;
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
/// Contract assertions work even when contracts are disabled:
/// \code  
/// cmake -DHPX_WITH_CONTRACTS=OFF  # HPX_CONTRACT_ASSERT still maps to HPX_ASSERT
/// \endcode
///
/// Optional: Unify assertion systems (when contracts enabled):
/// \code  
/// cmake -DHPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS=ON
/// \endcode

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>


#ifdef HPX_HAVE_CONTRACTS
    #if __cpp_contracts
        #define HPX_PRE(x) pre((x))
        #define HPX_CONTRACT_ASSERT(x) contract_assert((x))
        #define HPX_POST(x) post((x))


        #ifdef HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS 
            #define HPX_ASSERT(x) HPX_CONTRACT_ASSERT((x))
        #endif


    #else
        #pragma message("Warning: Contracts require C++26 or later. Falling back to HPX_ASSERT. 
            Until C++26 is available, HPX_PRE and HPX_POST are inside function bodies. 
            After migrating to C++26 they will be moved to function declarations thus HPX_PRE AND HPX_POST will not parse if contracts are not supported. ")        
        #define HPX_PRE(x) HPX_ASSERT((x))
        #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))  
        #define HPX_POST(x) HPX_ASSERT((x)) 
    #endif

#else
    #define HPX_PRE(x)
    #define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))
    #define HPX_POST(x)
#endif


