//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This must fail compiling

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <utility>

///////////////////////////////////////////////////////////////////////////////
int main()
{
    using hpx::lcos::local::spinlock;
    spinlock m1, m2(std::move(m1));
    return 0;
}
