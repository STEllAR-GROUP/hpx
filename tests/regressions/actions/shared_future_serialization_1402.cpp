//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/future.hpp>

struct movable_only
{
    movable_only() {}
    movable_only(movable_only&&) {}
};

void pass_shared_future_movable(hpx::shared_future<movable_only> const& obj) {}

HPX_PLAIN_ACTION(pass_shared_future_movable, pass_shared_future_movable_action)

int main() {
    pass_shared_future_movable_action act;
    hpx::shared_future<movable_only> arg =
        hpx::make_ready_future<movable_only>(movable_only());
    act(hpx::find_here(), arg);
    return 0;
}
