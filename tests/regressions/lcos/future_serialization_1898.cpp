//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include <hpx/util/lightweight_test.hpp>

struct test_server
    : hpx::components::component_base<test_server>
{
    test_server() { alive++; }
    ~test_server() { alive--; }

    static boost::atomic<int> alive;
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

boost::atomic<int> test_server::alive(0);

hpx::id_type test(hpx::future<hpx::id_type> fid)
{
    hpx::id_type id = fid.get();
    HPX_TEST(hpx::naming::detail::gid_was_split(id.get_gid()));
    return id;
}

HPX_PLAIN_ACTION(test);

int main()
{
    hpx::id_type loc = hpx::find_here();
    {
        HPX_TEST(test_server::alive == 0);
        hpx::id_type gid = hpx::new_<test_server>(loc).get();
        HPX_TEST(test_server::alive == 1);
//         HPX_TEST(!hpx::naming::detail::gid_was_split(gid.get_gid()));

        auto remote_localities = hpx::find_remote_localities();
        for(hpx::id_type loc : remote_localities)
        {
            {
                hpx::future<hpx::id_type> test_fid = hpx::make_ready_future(gid);
                hpx::future<hpx::id_type> fid
                    = hpx::async(test_action(), loc, std::move(test_fid));
                HPX_TEST(test_server::alive == 1);

                hpx::id_type new_gid = fid.get();
                HPX_TEST_NEQ(
                    hpx::naming::detail::get_credit_from_gid(gid.get_gid())
                  , hpx::naming::detail::get_credit_from_gid(new_gid.get_gid())
                );
            }

            {
                hpx::lcos::local::promise<hpx::id_type> pid;

                hpx::future<hpx::id_type> test_fid = pid.get_future();
                hpx::future<hpx::id_type> fid
                    = hpx::async(test_action(), loc, std::move(test_fid));
                HPX_TEST(test_server::alive == 1);

                hpx::this_thread::yield();

                pid.set_value(gid);
                HPX_TEST(test_server::alive == 1);

                hpx::id_type new_gid = fid.get();
                HPX_TEST_NEQ(
                    hpx::naming::detail::get_credit_from_gid(gid.get_gid())
                  , hpx::naming::detail::get_credit_from_gid(new_gid.get_gid())
                );
            }


            HPX_TEST(test_server::alive == 1);
        }
        HPX_TEST(test_server::alive == 1);
    }

    return 0;
}
