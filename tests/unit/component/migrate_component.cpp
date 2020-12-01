//  Copyright (c) 2014-2016 Hartmut Kaiser
//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct dummy_server : hpx::components::component_base<dummy_server>
{
    dummy_server() = default;

    hpx::id_type call() const { return hpx::find_here(); }

    HPX_DEFINE_COMPONENT_ACTION(dummy_server, call);
};

typedef hpx::components::component<dummy_server> dummy_server_type;
HPX_REGISTER_COMPONENT(dummy_server_type, dummy_server);

typedef dummy_server::call_action dummy_action;
HPX_REGISTER_ACTION(dummy_action);

struct dummy_client : hpx::components::client_base<dummy_client, dummy_server>
{
    typedef hpx::components::client_base<dummy_client, dummy_server> base_type;

    dummy_client() = default;

    dummy_client(hpx::id_type const& id)
      : base_type(id)
    {}
    dummy_client(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    hpx::id_type call() const
    {
        return hpx::async<dummy_action>(this->get_id()).get();
    }
};

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
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        return hpx::find_here();
    }

    void busy_work() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        hpx::this_thread::sleep_for(std::chrono::seconds(1));
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
    }

    hpx::future<void> lazy_busy_work() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = hpx::make_ready_future_after(std::chrono::seconds(1));

        return f.then(
            [this](hpx::future<void> && f) -> void
            {
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
                f.get();
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            });
    }

    int get_data() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        return data_;
    }

    hpx::future<int> lazy_get_data() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = hpx::make_ready_future(data_);

        return f.then(
            [this](hpx::future<int> && f) -> int
            {
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
                auto result = f.get();
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
                return result;
            });
    }

    dummy_client lazy_get_client(hpx::id_type there) const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = dummy_client(hpx::new_<dummy_server>(there));

        return f.then(
            [this](dummy_client && f) -> hpx::id_type
            {
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
                auto result = f.get();
                HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
                return result;
            });
    }

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // component's constructor.
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
    HPX_DEFINE_COMPONENT_ACTION(
        test_server, lazy_busy_work, lazy_busy_work_action);

    HPX_DEFINE_COMPONENT_ACTION(test_server, get_data, get_data_action);
    HPX_DEFINE_COMPONENT_ACTION(test_server, lazy_get_data, lazy_get_data_action);
    HPX_DEFINE_COMPONENT_ACTION(
        test_server, lazy_get_client, lazy_get_client_action);

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & data_;
        // clang-format on
    }

private:
    int data_;
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

typedef test_server::busy_work_action busy_work_action;
HPX_REGISTER_ACTION_DECLARATION(busy_work_action);
HPX_REGISTER_ACTION(busy_work_action);

typedef test_server::lazy_busy_work_action lazy_busy_work_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_busy_work_action);
HPX_REGISTER_ACTION(lazy_busy_work_action);

typedef test_server::get_data_action get_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_data_action);
HPX_REGISTER_ACTION(get_data_action);

typedef test_server::lazy_get_data_action lazy_get_data_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_get_data_action);
HPX_REGISTER_ACTION(lazy_get_data_action);

typedef test_server::lazy_get_client_action lazy_get_client_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_get_client_action);
HPX_REGISTER_ACTION(lazy_get_client_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}
    test_client(hpx::id_type && id) : base_type(std::move(id)) {}

    hpx::id_type call() const
    {
        return call_action()(this->get_id());
    }

    hpx::future<void> busy_work() const
    {
        return hpx::async<busy_work_action>(this->get_id());
    }

    hpx::future<void> lazy_busy_work() const
    {
        return hpx::async<lazy_busy_work_action>(this->get_id());
    }

    int get_data() const
    {
        return get_data_action()(this->get_id());
    }

    int lazy_get_data() const
    {
        return lazy_get_data_action()(this->get_id()).get();
    }

    dummy_client lazy_get_client(hpx::id_type there) const
    {
        return lazy_get_client_action()(this->get_id(), there);
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
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.get_data(), 42);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

bool test_migrate_lazy_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);

    try {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_data(), 42);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

bool test_migrate_lazy_component_client(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

    try {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_client(source).call(), source);
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
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        HPX_TEST_EQ(t1.get_data(), 42);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
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

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_lazy_busy_component(hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);

    // add some concurrent busy work
    hpx::future<void> lazy_busy_work = t1.lazy_busy_work();

    try {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        HPX_TEST_EQ(t1.lazy_get_data(), 42);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_data(), 42);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    // the busy work should be finished by now, wait anyways
    lazy_busy_work.wait();

    return true;
}

bool test_migrate_lazy_busy_component_client(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

    // add some concurrent busy work
    hpx::future<void> lazy_busy_work = t1.lazy_busy_work();

    try {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate(t1, target));

        HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_client(source).call(), source);
    }
    catch (hpx::exception const& e) {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    // the busy work should be finished by now, wait anyways
    lazy_busy_work.wait();

    return true;
}

////////////////////////////////////////////////////////////////////////////////
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
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate(t1, target));

            HPX_TEST_EQ(t1.get_data(), 42);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
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

bool test_migrate_lazy_component2(hpx::id_type source, hpx::id_type target)
{
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);

    std::size_t N = 100;

    try {
        // migrate an object back and forth between 2 localities a couple of
        // times
        for(std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate(t1, target));

            HPX_TEST_EQ(t1.lazy_get_data(), 42);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.lazy_get_data(), 42);

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

bool test_migrate_lazy_component_client2(hpx::id_type source, hpx::id_type target)
{
    test_client t1 = hpx::new_<test_client>(source, 42);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

    std::size_t N = 100;

    try {
        // migrate an object back and forth between 2 localities a couple of
        // times
        for(std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate(t1, target));

            HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.lazy_get_data(), 42);
            HPX_TEST_EQ(t2.lazy_get_client(source).call(), source);

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
                // migrate t1 to the target (loc2)
                test_client t2(hpx::components::migrate(t1, target));

                HPX_TEST_EQ(t1.get_data(), 42);

                // wait for migration to be done
                HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

                // the migrated object should have the same id as before
                HPX_TEST_EQ(t1.get_id(), t2.get_id());

                // the migrated object should live on the target now
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

    hpx::cout << std::endl;

    return true;
}

bool test_migrate_lazy_busy_component2(hpx::id_type source, hpx::id_type target)
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
                // migrate t1 to the target (loc2)
                test_client t2(hpx::components::migrate(t1, target));

                HPX_TEST_EQ(t1.lazy_get_data(), 42);

                // wait for migration to be done
                HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

                // the migrated object should have the same id as before
                HPX_TEST_EQ(t1.get_id(), t2.get_id());

                // the migrated object should live on the target now
                HPX_TEST_EQ(t2.call(), target);
                HPX_TEST_EQ(t2.lazy_get_data(), 42);

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
                HPX_TEST_EQ(t1.lazy_get_data(), 42);
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

    hpx::cout << std::endl;

    return true;
}

bool test_migrate_lazy_busy_component_client2(
    hpx::id_type source, hpx::id_type target)
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
                // migrate t1 to the target (loc2)
                test_client t2(hpx::components::migrate(t1, target));

                HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);

                // wait for migration to be done
                HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

                // the migrated object should have the same id as before
                HPX_TEST_EQ(t1.get_id(), t2.get_id());

                // the migrated object should live on the target now
                HPX_TEST_EQ(t2.call(), target);
                HPX_TEST_EQ(t2.lazy_get_client(source).call(), source);

                hpx::cout << "." << std::flush;

                std::swap(source, target);
            }

            hpx::cout << std::endl;
        }
    );

    // Second, we generate tons of work which should automatically follow
    // the migration.
    hpx::future<void> create_work = hpx::async(
        [t1, N, target]()
        {
            for(std::size_t i = 0; i < 2*N; ++i)
            {
                hpx::cout
                    << hpx::naming::get_locality_id_from_id(t1.call())
                    << std::flush;
                HPX_TEST_EQ(t1.lazy_get_client(target).call(), target);
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

    hpx::cout << std::endl;

    return true;
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_component(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_component_client: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component_client(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_component_client: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component_client(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_busy_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_busy_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_busy_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_busy_component(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_busy_component: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_busy_component: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_busy_component_client: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component_client(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_busy_component_client: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component_client(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_component2(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component2(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_component_client2: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component_client2(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_component_client2: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_component_client2(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test_migrate_busy_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_busy_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_busy_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_busy_component2(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_busy_component2: ->" << id << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component2(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_busy_component2: <-" << id << std::endl;
        HPX_TEST(test_migrate_lazy_busy_component2(id, hpx::find_here()));

        hpx::cout << "test_migrate_lazy_busy_component_client2: ->" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_busy_component_client2(hpx::find_here(), id));
        hpx::cout << "test_migrate_lazy_busy_component_client2: <-" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_busy_component_client2(id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}
#endif
