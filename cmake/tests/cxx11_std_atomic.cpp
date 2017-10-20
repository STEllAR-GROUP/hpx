////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <atomic>

int main()
{
    std::atomic_flag af = ATOMIC_FLAG_INIT;
    if (af.test_and_set())
        af.clear();

    std::atomic<int> ai;
    ai.store(0);
    int i = ai.load();

    std::memory_order mo;
    mo = std::memory_order_relaxed;
    mo = std::memory_order_acquire;
    mo = std::memory_order_release;
    mo = std::memory_order_acq_rel;
    mo = std::memory_order_seq_cst;
}
