////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <atomic>
#include <cstdint>

template <typename T>
void test_atomic()
{
    std::atomic<T> a;
    a.store(T{});
    [[maybe_unused]] T i = a.load();

    // force using libatomic, if needed
    [[maybe_unused]] bool b = a.is_lock_free();

    // Newer HIP environments seem to need libatomic for compare_exchange
    T expected;
    a.compare_exchange_strong(expected, i);
}

struct index_data
{
    std::uint16_t first;
    std::uint16_t second;
};

int main()
{
// ATOMIC_FLAG_INIT is deprecated starting C++20
#if defined(HPX_HAVE_CXX11_ATOMIC_FLAG_INIT)
    std::atomic_flag af = ATOMIC_FLAG_INIT;
#else
    std::atomic_flag af;
#endif
    if (af.test_and_set())
        af.clear();

    test_atomic<int>();
    test_atomic<std::uint8_t>();
    test_atomic<std::uint16_t>();
    test_atomic<std::uint32_t>();
    test_atomic<std::uint64_t>();
    test_atomic<index_data>();

    std::memory_order mo;
    mo = std::memory_order_relaxed;
    mo = std::memory_order_acquire;
    mo = std::memory_order_release;
    mo = std::memory_order_acq_rel;
    mo = std::memory_order_seq_cst;
    (void) mo;
}
