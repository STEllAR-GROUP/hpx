//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// verify #1405 is fixed (Allow component constructors to take movable
// only types)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
struct moveonly
{
private:
    HPX_MOVABLE_BUT_NOT_COPYABLE(moveonly);

public:
    moveonly() {}

    moveonly(moveonly&&) {}
    moveonly& operator=(moveonly&&) { return *this; }
};

struct moveable
{
    moveable() {}

    moveable(moveable&&) {}
    moveable& operator=(moveable&&) { return *this; }

    moveable(moveable const&) {}
    moveable& operator=(moveable const&) { return *this; }
};

boost::atomic<int> constructed_from_moveonly(0);
boost::atomic<int> constructed_from_moveable(0);

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::managed_component_base<test_server>
{
    test_server() {}

    test_server(moveonly && arg)
    {
        ++constructed_from_moveonly;
    }
    test_server(moveable const& arg)
    {
        ++constructed_from_moveable;
    }
};

typedef hpx::components::managed_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char** argv_init)
{
    hpx::new_<test_server>(hpx::find_here(), moveonly()).get();
    HPX_TEST_EQ(constructed_from_moveonly.load(), 1);

    moveable o;
    hpx::new_<test_server>(hpx::find_here(), o).get();
    HPX_TEST_EQ(constructed_from_moveable.load(), 1);

    hpx::new_<test_server>(hpx::find_here(), moveable()).get();
    HPX_TEST_EQ(constructed_from_moveable.load(), 2);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
