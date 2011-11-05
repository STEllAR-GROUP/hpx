
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef q35025ejfwdfho48yt53thgworvgnsoncq83rhew98fnhlacnmamcpeqfmpcdoivonwqnf
#define q35025ejfwdfho48yt53thgworvgnsoncq83rhew98fnhlacnmamcpeqfmpcdoivonwqnf

#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/components/client_base.hpp>

using hpx::cout;
using hpx::flush;
using hpx::find_here;

using hpx::components::managed_component_base;
using hpx::components::simple_component_base;
using hpx::components::stubs::stub_base;
using hpx::components::client_base;

//HPX_REGISTER_COMPONENT_MODULE();

namespace server
{
    struct test_component
        : managed_component_base<test_component>
    {
        test_component()
        {
            cout << "server::test_component::test_component()\n" << flush;
        }

        void finalize()
        {
            cout << "server::test_component::finalize()\n" << flush;
        }

        ~test_component()
        {
            cout << "~server::test_component::test_component()\n" << flush;
        }
    };
}
namespace stubs
{
        struct test_component
            : stub_base<
                server::test_component
            >
        {
        };
}
struct test_component
    : client_base<test_component, stubs::test_component>
{
    typedef client_base<test_component, stubs::test_component> base_type;
    test_component()
        : base_type(stubs::test_component::create_sync(find_here()))
    {
        cout << "test_component::test_component()\n" << flush;
    }

    ~test_component()
    {
        cout << "~test_component::test_component()\n" << flush;
    }
};

namespace server
{
    struct simple_test_component
        : simple_component_base<simple_test_component>
    {
        simple_test_component()
        {
            cout << "server::simple_test_component::simple_test_component()\n" << flush;
        }

        void finalize()
        {
            cout << "server::simple_test_component::finalize()\n" << flush;
        }

        ~simple_test_component()
        {
            cout << "~server::simple_test_component::simple_test_component()\n" << flush;
        }
    };
}
namespace stubs
{
        struct simple_test_component
            : stub_base<
                server::simple_test_component
            >
        {
        };
}
struct simple_test_component
    : client_base<simple_test_component, stubs::simple_test_component>
{
    typedef client_base<simple_test_component, stubs::simple_test_component> base_type;
    simple_test_component()
        : base_type(stubs::simple_test_component::create_sync(find_here()))
    {
        cout << "simple_test_component::simple_test_component()\n" << flush;
    }

    ~simple_test_component()
    {
        cout << "~simple_test_component::simpl_test_component()\n" << flush;
    }
};

#endif
