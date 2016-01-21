//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::migration_support<
        hpx::components::simple_component_base<test_server>
    >
{
    test_server() {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized  is moved into the
    // components constructor.
    test_server(test_server const& rhs) {}
    test_server(test_server && rhs) {}

    test_server& operator=(test_server const &) { return *this; }
    test_server& operator=(test_server &&) { return *this; }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);

    template <typename Archive>
    void serialize(Archive&ar, unsigned version) {}

private:
    ;
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

    hpx::id_type call() const
    {
        return call_action()(this->get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);

    try {
        // migrate of t1 to the target
        test_client t2(hpx::components::migrate(t1, target));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should life on the target now
        HPX_TEST_EQ(t2.call(), target);

        return true;
    }
    catch (hpx::exception const&) {
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////////
struct busy_test_server
  : hpx::components::migration_support<
        hpx::components::simple_component_base<busy_test_server>
    >
{
    busy_test_server()
      : latch_(0)
    {}
    busy_test_server(std::ptrdiff_t count)
      : latch_(new hpx::lcos::local::latch(count))
    {}
    ~busy_test_server()
    {
        delete latch_;
    }

    void count_down_and_wait()
    {
        HPX_ASSERT(0 != latch_);
        latch_->count_down_and_wait();

        hpx::this_thread::sleep_for(boost::chrono::seconds(1));
    }

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // components constructor.
    busy_test_server(busy_test_server const& rhs)
      : latch_(0)
    {}
    busy_test_server(busy_test_server && rhs)
      : latch_(0)
    {}

    busy_test_server& operator=(busy_test_server const &) { return *this; }
    busy_test_server& operator=(busy_test_server &&) { return *this; }

    HPX_DEFINE_COMPONENT_ACTION(busy_test_server, count_down_and_wait);
    HPX_DEFINE_COMPONENT_ACTION(busy_test_server, call);

    template <typename Archive>
    void serialize(Archive&ar, unsigned version) {}

private:
    hpx::lcos::local::latch* latch_;
};

typedef hpx::components::simple_component<busy_test_server> busy_server_type;
HPX_REGISTER_COMPONENT(busy_server_type, busy_test_server);

typedef busy_test_server::call_action busy_call_action;
HPX_REGISTER_ACTION_DECLARATION(busy_call_action);
HPX_REGISTER_ACTION(busy_call_action);

typedef busy_test_server::count_down_and_wait_action count_down_and_wait_action;
HPX_REGISTER_ACTION_DECLARATION(count_down_and_wait_action);
HPX_REGISTER_ACTION(count_down_and_wait_action);

struct busy_test_client
  : hpx::components::client_base<busy_test_client, busy_test_server>
{
    typedef hpx::components::client_base<busy_test_client, busy_test_server>
        base_type;

    busy_test_client() {}
    busy_test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

    hpx::id_type call() const
    {
        return busy_call_action()(this->get_id());
    }

    hpx::future<void> count_down_and_wait()
    {
        return hpx::async<count_down_and_wait_action>(this->get_id());
    }
};

bool test_migrate_busy_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    busy_test_client t1 = hpx::new_<busy_test_client>(source, 2);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    std::vector<hpx::future<void> > busy_work;
    busy_work.reserve(2);

    // add some busy work
//    for (std::size_t i = 0; i != 10; ++i)
    busy_work.push_back(t1.count_down_and_wait());

    try {
        // migrate of t1 to the target
        busy_test_client t2(hpx::components::migrate(t1, target));

        // add more busy work
//        for (std::size_t i = 0; i != 10; ++i)
        busy_work.push_back(t1.count_down_and_wait());

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should life on the target now
        HPX_TEST_EQ(t2.call(), target);
    }
    catch (hpx::exception const&) {
        return false;
    }

    hpx::wait_all(busy_work);

    return true;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(test_migrate_component(hpx::find_here(), id));
        HPX_TEST(test_migrate_component(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(test_migrate_busy_component(hpx::find_here(), id));
        HPX_TEST(test_migrate_busy_component(id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}

