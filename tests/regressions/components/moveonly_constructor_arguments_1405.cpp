//  Copyright (c) 2007-2015 Hartmut Kaiser
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

struct copyonly
{
    copyonly() {}

    copyonly(copyonly const&) {}
    copyonly& operator=(copyonly const&) { return *this; }
};

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::managed_component_base<test_server>
{
    test_server() {}
    test_server(moveonly&& arg) {}
    test_server(copyonly const& arg) {}
};

typedef hpx::components::managed_component<test_server> server_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_type, test_server);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char** argv_init)
{
    hpx::new_<test_server>(hpx::find_here(), moveonly()).get();

    copyonly co;
    hpx::new_<test_server>(hpx::find_here(), co).get();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
