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
/// behavior when contracts are not available.
///
/// ## API Reference:
/// - **HPX_PRE(condition)**: Precondition contracts (no-op in fallback mode)
/// - **HPX_POST(condition)**: Postcondition contracts (no-op in fallback mode)
/// - **HPX_CONTRACT_ASSERT(condition)**: Contract assertions (always available, maps to HPX_ASSERT)
///
/// ## Configuration:
/// Enable with: `cmake -DHPX_WITH_CONTRACTS=ON -DCMAKE_CXX_STANDARD=26`
///
/// See docs/index.rst for comprehensive usage guide.

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>

// Contract implementation: automatically selects native C++26 contracts
// or provides appropriate fallback behavior based on compiler capabilities
#if defined(HPX_HAVE_CXX26_CONTRACTS)

// Native C++26 contracts mode
#define HPX_PRE(x) pre(x)
#define HPX_CONTRACT_ASSERT(x) contract_assert(x)
#define HPX_POST(x) post(x)

#if defined(HPX_CONTRACTS_HAVE_ASSERTS_AS_CONTRACT_ASSERTS)
// Override HPX_ASSERT to use contract assertions
#define HPX_ASSERT(x) contract_assert(x)
#endif

#else

// Fallback mode: PRE/POST become no-ops for forward compatibility,
// CONTRACT_ASSERT maps to HPX_ASSERT for runtime validation
#define HPX_PRE(x)
#define HPX_CONTRACT_ASSERT(x) HPX_ASSERT((x))
#define HPX_POST(x)

#endif
