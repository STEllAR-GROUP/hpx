//  Copyright (c) 2014-2016 Hartmut Kaiser
//  Copyright (c)      2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/chrono.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::migration_support<
        hpx::components::component_base<test_server>
    >
{
    typedef hpx::components::migration_support<
            hpx::components::component_base<test_server>
        > base_type;

    test_server(int data = 0) : data_(data) {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    void busy_work() const
    {
        hpx::this_thread::sleep_for(boost::chrono::seconds(1));
    }

    int get_data() const
    {
        return data_;
    }

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // components constructor.
    test_server(test_server const& rhs)
      : base_type(rhs), data_(rhs.data_)
    {}

    test_server(test_server && rhs)
      : base_type(std::move(rhs)), data_(rhs.data_)
    {}

    test_server& operator=(test_server const & rhs)
    {
        data_ = rhs.data_;
        return *this;
    }
    test_server& operator=(test_server && rhs)
    {
        data_ = rhs.data_;
        return *this;
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, call, call_action);
    HPX_DEFINE_COMPONENT_ACTION(test_server, busy_work, busy_work_action);
    HPX_DEFINE_COMPONENT_ACTION(test_server, get_data, get_data_action);

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        ar & data_;
    }

private:
    int data_;
};

typedef hpx::components::simple_component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

typedef test_server::busy_work_action busy_work_action;
HPX_REGISTER_ACTION_DECLARATION(busy_work_action);
HPX_REGISTER_ACTION(busy_work_action);

typedef test_server::get_data_action get_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_data_action);
HPX_REGISTER_ACTION(get_data_action);

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

    hpx::future<void> busy_work() const
    {
        return hpx::async<busy_work_action>(this->get_id());
    }

    int get_data() const
    {
        return get_data_action()(this->get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);

    try {
        // migrate of t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should life on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.get_data(), 42);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_busy_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);

    // add some concurrent busy work
    hpx::future<void> busy_work = t1.busy_work();

    try {
        // migrate of t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        HPX_TEST_EQ(t1.get_data(), 42);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should life on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.get_data(), 42);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    // the busy work should be finished by now, wait anyways
    busy_work.wait();

    return true;
}

bool test_migrate_component2(hpx::id_type source, hpx::id_type target)
{
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);

    std::size_t N = 100;

    try {
        // migrate an object back and forth between 2 localities a couple of
        // times
        for(std::size_t i = 0; i < N; ++i)
        {
            // migrate of t1 to the target (loc2)
            test_client t2(hpx::components::migrate(t1, target));

            HPX_TEST_EQ(t1.get_data(), 42);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should life on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.get_data(), 42);

            hpx::cout << "." << std::flush;

            std::swap(source, target);
        }

        hpx::cout << std::endl;
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

bool test_migrate_busy_component2(hpx::id_type source, hpx::id_type target)
{
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);

    std::size_t N = 100;

    // First, spawn a thread (locally) that will migrate the object between
    // source and target a few times
    hpx::future<void> migrate_future = hpx::async(
        [source, target, t1, N]() mutable
        {
            for(std::size_t i = 0; i < N; ++i)
            {
                // migrate of t1 to the target (loc2)
                test_client t2(hpx::components::migrate(t1, target));

                HPX_TEST_EQ(t1.get_data(), 42);

                // wait for migration to be done
                HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

                // the migrated object should have the same id as before
                HPX_TEST_EQ(t1.get_id(), t2.get_id());

                // the migrated object should life on the target now
                HPX_TEST_EQ(t2.call(), target);
                HPX_TEST_EQ(t2.get_data(), 42);

                hpx::cout << "." << std::flush;

                std::swap(source, target);
            }

            hpx::cout << std::endl;
        }
    );

    // Second, we generate tons of work which should automatically follow
    // the migration.
    hpx::future<void> create_work = hpx::async(
        [t1, N]()
        {
            for(std::size_t i = 0; i < 2*N; ++i)
            {
                hpx::cout
                    << hpx::naming::get_locality_id_from_id(t1.call())
                    << std::flush;
                HPX_TEST_EQ(t1.get_data(), 42);
            }
        }
    );

    hpx::wait_all(migrate_future, create_work);

    // rethrow exceptions
    try {
        migrate_future.get();
        create_work.get();
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_component(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_busy_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_busy_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_busy_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_busy_component(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_component2(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_busy_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_busy_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_busy_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_busy_component2(id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}

