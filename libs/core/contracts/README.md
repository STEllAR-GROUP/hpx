<!--
    Copyright (c) 2025 The STE||AR-Group
    Copyright (c) 2025 Alexandros Papadakis
    Copyright (c) 2025 Panagiotis Syskakis

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
-->

# HPX Contracts Module

A forward-compatible C++ contracts implementation for HPX with intelligent fallback to assertions.

## Quick Start

```cpp
#include <hpx/contracts.hpp>

// Preconditions and postconditions with return value access
int factorial(int n) HPX_PRE(n >= 0) HPX_POST(r: r > 0) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

// Contract assertions (always available)
void process(std::vector<int>& data, size_t index) {
    HPX_CONTRACT_ASSERT(index < data.size());
    data[index] *= 2;
}
```

## Build Configuration

```bash
# Enable contracts
cmake -DHPX_WITH_CONTRACTS=ON -DCMAKE_CXX_STANDARD=26

# Enable contract-enhanced assertions (optional)
cmake -DHPX_WITH_CONTRACTS=ON -DHPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS=ON

# Contracts disabled (hpx_contract_assert will work as hpx_assert)
cmake -DHPX_WITH_CONTRACTS=OFF #default
```

## Advanced Features

### Contract-Enhanced Assertions
When `HPX_HAVE_ASSERTS_AS_CONTRACT_ASSERTS=ON`, regular `HPX_ASSERT` calls are automatically upgraded to use contract assertions in C++26 mode, providing enhanced contract semantics throughout your codebase.

## API Reference

- **`HPX_PRE(condition)`**: Precondition contracts
- **`HPX_POST(condition)`**: Postcondition contracts
- **`HPX_CONTRACT_ASSERT(condition)`**: Contract assertions (always available)

## Documentation

See [Module Documentation](docs/index.rst) for comprehensive usage guide, API reference, and implementation details.

## Features

- ✅ Automatic C++26 native contract detection
- ✅ Graceful fallback to HPX_ASSERT 
- ✅ Zero overhead when disabled
- ✅ Forward-compatible API
- ✅ Comprehensive test suite with automatic mode detection