//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct decrement_server
  : hpx::components::managed_component_base<decrement_server>
{
    boost::int32_t call(boost::int32_t i) const
    {
        return i - 1;
    }

    HPX_DEFINE_COMPONENT_ACTION(decrement_server, call);
};

typedef hpx::components::managed_component<decrement_server> server_type;
HPX_REGISTER_COMPONENT(server_type, decrement_server);

typedef decrement_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<int> callback_called(0);

void cb(boost::system::error_code const& ec,
    hpx::parcelset::parcel const& p)
{
    ++callback_called;
}

///////////////////////////////////////////////////////////////////////////////
void test_remote_async_cb(hpx::id_type const& target)
{
    typedef hpx::components::client<decrement_server> decrement_client;

    {
        decrement_client dec_f =
            hpx::components::new_<decrement_client>(target);

        call_action call;

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 = hpx::async_cb(call, dec_f, &cb, 42);
        HPX_TEST_EQ(f1.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb(hpx::launch::all, call, dec_f, &cb, 42);
        HPX_TEST_EQ(f2.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    {
        decrement_client dec_f =
            hpx::components::new_<decrement_client>(target);
        hpx::id_type dec = dec_f.get_id();

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 =
            hpx::async_cb<call_action>(dec_f, &cb, 42);
        HPX_TEST_EQ(f1.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb<call_action>(hpx::launch::all, dec_f, &cb, 42);
        HPX_TEST_EQ(f2.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);
    }
}

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& id : localities)
    {
        test_remote_async_cb(id);
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

