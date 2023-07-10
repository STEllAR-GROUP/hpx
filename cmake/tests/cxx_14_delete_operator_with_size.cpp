//  Copyright (c) 2023 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstddef>
#include <memory>

/// clang fails to compile this with libstdc++ even though gcc does fine

int main()
{
    using dealloc_fn = void (*)(void*, std::size_t);
    dealloc_fn const dealloc = [](void* const p, std::size_t const s) {
        ::operator delete[](p, s + sizeof(std::size_t) + sizeof(dealloc_fn));
    };
    (void) dealloc;
}
