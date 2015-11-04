//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/traits.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::int32_t increment(boost::int32_t i)
{
    return i + 1;
}
HPX_PLAIN_ACTION(increment);

boost::int32_t increment_with_future(hpx::shared_future<boost::int32_t> fi)
{
    return fi.get() + 1;
}
HPX_PLAIN_ACTION(increment_with_future);

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
struct test_server : hpx::components::simple_component_base<test_server>
{
};

typedef hpx::components::simple_component<test_server> test_server_type;
HPX_REGISTER_COMPONENT(test_server_type, test_server);

struct test_client : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server> base_type;

    test_client(hpx::id_type const& id)
      : base_type(id)
    {}
    test_client(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}
};

///////////////////////////////////////////////////////////////////////////////
boost::atomic<int> callback_called(0);

void cb(boost::system::error_code const& ec,
    hpx::parcelset::parcel const& p)
{
    ++callback_called;
}

///////////////////////////////////////////////////////////////////////////////
void test_remote_async_cb_colocated(test_client const& target)
{
    {
        increment_action inc;

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 =
            hpx::async_cb(inc, hpx::colocated(target), &cb, 42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb(hpx::launch::all, inc, hpx::colocated(target), &cb, 42);
        HPX_TEST_EQ(f2.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    {
        increment_with_future_action inc;

        hpx::promise<boost::int32_t> p;
        hpx::shared_future<boost::int32_t> f = p.get_future();

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 =
            hpx::async_cb(inc, hpx::colocated(target), &cb, f);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb(hpx::launch::all, inc, hpx::colocated(target), &cb, f);

        p.set_value(42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(f2.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 2);
    }

    {
        callback_called.store(0);
        hpx::future<boost::int32_t> f1 =
            hpx::async_cb<increment_action>(hpx::colocated(target), &cb, 42);
        HPX_TEST_EQ(f1.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 = hpx::async_cb<increment_action>(
            hpx::launch::all, hpx::colocated(target), &cb, 42);
        HPX_TEST_EQ(f2.get(), 43);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(hpx::colocated(target));
        hpx::id_type dec = dec_f.get();

        call_action call;

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 = hpx::async_cb(call, dec, &cb, 42);
        HPX_TEST_EQ(f1.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb(hpx::launch::all, call, dec, &cb, 42);
        HPX_TEST_EQ(f2.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    {
        hpx::future<hpx::id_type> dec_f =
            hpx::components::new_<decrement_server>(hpx::colocated(target));
        hpx::id_type dec = dec_f.get();

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 =
            hpx::async_cb<call_action>(dec, &cb, 42);
        HPX_TEST_EQ(f1.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);

        callback_called.store(0);
        hpx::future<boost::int32_t> f2 =
            hpx::async_cb<call_action>(hpx::launch::all, dec, &cb, 42);
        HPX_TEST_EQ(f2.get(), 41);
        HPX_TEST_EQ(callback_called.load(), 1);
    }

    {
        increment_with_future_action inc;
        hpx::shared_future<boost::int32_t> f =
            hpx::async(hpx::launch::deferred, hpx::util::bind(&increment, 42));

        callback_called.store(0);
        hpx::future<boost::int32_t> f1 = hpx::async_cb(
            inc, hpx::colocated(target), &cb, f);
        hpx::future<boost::int32_t> f2 = hpx::async_cb(
            hpx::launch::all, inc, hpx::colocated(target), &cb, f);

        HPX_TEST_EQ(f1.get(), 44);
        HPX_TEST_EQ(f2.get(), 44);
        HPX_TEST_EQ(callback_called.load(), 2);
    }
}

int hpx_main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    for (hpx::id_type const& id : localities)
    {
        test_client client(hpx::new_<test_client>(id));
        test_remote_async_cb_colocated(client);
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

