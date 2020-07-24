//  Copyright (c) 2019 Maximilian Bremer
//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct test_server_base
  : hpx::components::abstract_base_migration_support<
        hpx::components::abstract_component_base<test_server_base>>
{
    test_server_base(int base_data = 0)
      : base_data_(base_data)
    {
    }
    virtual ~test_server_base() = default;

    hpx::id_type call() const
    {
        return hpx::find_here();
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server_base, call, call_action);

    void busy_work() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        hpx::this_thread::sleep_for(std::chrono::seconds(1));
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server_base, busy_work, busy_work_action);

    hpx::future<void> lazy_busy_work() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = hpx::make_ready_future_after(std::chrono::seconds(1));

        return f.then([this](hpx::future<void>&& f) -> void {
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            f.get();
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        });
    }
    HPX_DEFINE_COMPONENT_ACTION(
        test_server_base, lazy_busy_work, lazy_busy_work_action);

    int get_base_data() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        return base_data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(
        test_server_base, get_base_data, get_base_data_action);

    hpx::future<int> lazy_get_base_data() const
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = hpx::make_ready_future(base_data_);

        return f.then([this](hpx::future<int>&& f) -> int {
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            auto result = f.get();
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            return result;
        });
    }
    HPX_DEFINE_COMPONENT_ACTION(
        test_server_base, lazy_get_base_data, lazy_get_base_data_action);

    virtual int get_data() const
    {
        return get_base_data();
    }
    int get_data_nonvirt() const
    {
        return get_data();
    }
    HPX_DEFINE_COMPONENT_ACTION(
        test_server_base, get_data_nonvirt, get_data_action);

    virtual hpx::future<int> lazy_get_data() const
    {
        return lazy_get_base_data();
    }
    hpx::future<int> lazy_get_data_nonvirt() const
    {
        return lazy_get_data();
    }
    HPX_DEFINE_COMPONENT_ACTION(
        test_server_base, lazy_get_data_nonvirt, lazy_get_data_action);

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // component's constructor.
    test_server_base(test_server_base&& rhs)
      : base_data_(std::move(rhs.base_data_))
    {
    }

    test_server_base& operator=(test_server_base&& rhs)
    {
        base_data_ = std::move(rhs.base_data_);
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        ar & base_data_;
    }

private:
    int base_data_;
};

HPX_DEFINE_GET_COMPONENT_TYPE(test_server_base);

typedef test_server_base::call_action call_action;
HPX_REGISTER_ACTION_DECLARATION(call_action);
HPX_REGISTER_ACTION(call_action);

typedef test_server_base::busy_work_action busy_work_action;
HPX_REGISTER_ACTION_DECLARATION(busy_work_action);
HPX_REGISTER_ACTION(busy_work_action);

typedef test_server_base::lazy_busy_work_action lazy_busy_work_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_busy_work_action);
HPX_REGISTER_ACTION(lazy_busy_work_action);

typedef test_server_base::get_base_data_action get_base_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_base_data_action);
HPX_REGISTER_ACTION(get_base_data_action);

typedef test_server_base::lazy_get_base_data_action lazy_get_base_data_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_get_base_data_action);
HPX_REGISTER_ACTION(lazy_get_base_data_action);

typedef test_server_base::get_data_action get_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_data_action);
HPX_REGISTER_ACTION(get_data_action);

typedef test_server_base::lazy_get_data_action lazy_get_data_action;
HPX_REGISTER_ACTION_DECLARATION(lazy_get_data_action);
HPX_REGISTER_ACTION(lazy_get_data_action);

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::abstract_migration_support<
        hpx::components::component_base<test_server>, test_server_base>
{
    using base_type = hpx::components::abstract_migration_support<
        hpx::components::component_base<test_server>, test_server_base>;

    test_server(int base_data = 0, int data = 0)
      : base_type(base_data)
      , data_(data)
    {
    }

    int get_data() const override
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
        return data_;
    }

    hpx::future<int> lazy_get_data() const override
    {
        HPX_TEST_NEQ(pin_count(), std::uint32_t(0));

        auto f = hpx::make_ready_future(data_);

        return f.then([this](hpx::future<int>&& f) -> int {
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            auto result = f.get();
            HPX_TEST_NEQ(pin_count(), std::uint32_t(0));
            return result;
        });
    }

    // Components that should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // component's constructor.
    test_server(test_server&& rhs)
      : base_type(std::move(rhs))
      , data_(rhs.data_)
    {
    }

    test_server& operator=(test_server&& rhs)
    {
        this->test_server_base::operator=(
            std::move(static_cast<test_server_base&>(rhs)));
        data_ = rhs.data_;
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        ar & hpx::serialization::base_object<test_server_base>(*this);
        ar & data_;
    }

private:
    int data_;
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, test_server,
    "test_server_base");

///////////////////////////////////////////////////////////////////////////////
struct test_client : hpx::components::client_base<test_client, test_server_base>
{
    using base_type =
        hpx::components::client_base<test_client, test_server_base>;

    test_client() = default;
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}
    test_client(hpx::id_type&& id) : base_type(std::move(id)) {}

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

    int get_base_data() const
    {
        return get_base_data_action()(this->get_id());
    }

    int lazy_get_base_data() const
    {
        return lazy_get_base_data_action()(this->get_id()).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_polymorphic_component(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);
    HPX_TEST_EQ(t1.get_base_data(), 7);

    try
    {
        hpx::cout << "Migrating..." << hpx::endl;
        // migrate t1 to the target
        test_client t2(hpx::components::migrate<test_server>(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());
        hpx::cout << "...completed migrating..." << hpx::endl;

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.get_data(), 42);
        HPX_TEST_EQ(t2.get_base_data(), 7);

        hpx::cout << "...pass all tests!" << hpx::endl;
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_lazy_polymorphic_component(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);
    HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

    try
    {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate<test_server>(t1, target));

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_data(), 42);
        HPX_TEST_EQ(t2.lazy_get_base_data(), 7);
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_busy_polymorphic_component(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);
    HPX_TEST_EQ(t1.get_base_data(), 7);

    // add some concurrent busy work
    hpx::future<void> busy_work = t1.busy_work();

    try
    {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate<test_server>(t1, target));

        HPX_TEST_EQ(t1.get_data(), 42);
        HPX_TEST_EQ(t1.get_base_data(), 7);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.get_data(), 42);
        HPX_TEST_EQ(t2.get_base_data(), 7);
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    // the busy work should be finished by now, wait anyways
    busy_work.wait();

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_lazy_busy_polymorphic_component(
    hpx::id_type source, hpx::id_type target)
{
    // create component on given locality
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);
    HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

    // add some concurrent busy work
    hpx::future<void> lazy_busy_work = t1.lazy_busy_work();

    try
    {
        // migrate t1 to the target
        test_client t2(hpx::components::migrate<test_server>(t1, target));

        HPX_TEST_EQ(t1.lazy_get_data(), 42);
        HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

        // wait for migration to be done
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the migrated object should have the same id as before
        HPX_TEST_EQ(t1.get_id(), t2.get_id());

        // the migrated object should live on the target now
        HPX_TEST_EQ(t2.call(), target);
        HPX_TEST_EQ(t2.lazy_get_data(), 42);
        HPX_TEST_EQ(t2.lazy_get_base_data(), 7);
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    // the busy work should be finished by now, wait anyways
    lazy_busy_work.wait();

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_polymorphic_component2(
    hpx::id_type source, hpx::id_type target)
{
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);
    HPX_TEST_EQ(t1.get_base_data(), 7);

    std::size_t N = 100;

    try
    {
        // migrate an object back and forth between 2 localities a couple of
        // times
        for (std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate<test_server>(t1, target));

            HPX_TEST_EQ(t1.get_data(), 42);
            HPX_TEST_EQ(t1.get_base_data(), 7);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.get_data(), 42);
            HPX_TEST_EQ(t2.get_base_data(), 7);

            hpx::cout << "." << std::flush;

            std::swap(source, target);
        }

        hpx::cout << std::endl;
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

bool test_migrate_lazy_polymorphic_component2(
    hpx::id_type source, hpx::id_type target)
{
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.lazy_get_data(), 42);
    HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

    std::size_t N = 100;

    try
    {
        // migrate an object back and forth between 2 localities a couple of
        // times
        for (std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate<test_server>(t1, target));

            HPX_TEST_EQ(t1.lazy_get_data(), 42);
            HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.lazy_get_data(), 42);
            HPX_TEST_EQ(t2.lazy_get_base_data(), 7);

            hpx::cout << "." << std::flush;

            std::swap(source, target);
        }

        hpx::cout << std::endl;
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
bool test_migrate_busy_polymorphic_component2(
    hpx::id_type source, hpx::id_type target)
{
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);
    HPX_TEST_EQ(t1.get_base_data(), 7);

    std::size_t N = 100;

    // First, spawn a thread (locally) that will migrate the object between
    // source and target a few times
    hpx::future<void> migrate_future = hpx::async([source, target, t1,
                                                      N]() mutable {
        for (std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate<test_server>(t1, target));

            HPX_TEST_EQ(t1.get_data(), 42);
            HPX_TEST_EQ(t1.get_base_data(), 7);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.get_data(), 42);
            HPX_TEST_EQ(t2.get_base_data(), 7);

            hpx::cout << "." << std::flush;

            std::swap(source, target);
        }

        hpx::cout << std::endl;
    });

    // Second, we generate tons of work which should automatically follow
    // the migration.
    hpx::future<void> create_work = hpx::async([t1, N]() {
        for (std::size_t i = 0; i < 2 * N; ++i)
        {
            hpx::cout << hpx::naming::get_locality_id_from_id(t1.call())
                      << std::flush;
            HPX_TEST_EQ(t1.get_data(), 42);
            HPX_TEST_EQ(t1.get_base_data(), 7);
        }
    });

    hpx::wait_all(migrate_future, create_work);

    // rethrow exceptions
    try
    {
        migrate_future.get();
        create_work.get();
    }
    catch (hpx::exception const& e)
    {
        hpx::cout << hpx::get_error_what(e) << std::endl;
        return false;
    }

    return true;
}

bool test_migrate_lazy_busy_polymorphic_component2(
    hpx::id_type source, hpx::id_type target)
{
    test_client t1(hpx::new_<test_server>(source, 7, 42));
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    // the new object should live on the source locality
    HPX_TEST_EQ(t1.call(), source);
    HPX_TEST_EQ(t1.get_data(), 42);
    HPX_TEST_EQ(t1.get_base_data(), 7);

    std::size_t N = 100;

    // First, spawn a thread (locally) that will migrate the object between
    // source and target a few times
    hpx::future<void> migrate_future = hpx::async([source, target, t1,
                                                      N]() mutable {
        for (std::size_t i = 0; i < N; ++i)
        {
            // migrate t1 to the target (loc2)
            test_client t2(hpx::components::migrate<test_server>(t1, target));

            HPX_TEST_EQ(t1.lazy_get_data(), 42);
            HPX_TEST_EQ(t1.lazy_get_base_data(), 7);

            // wait for migration to be done
            HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

            // the migrated object should have the same id as before
            HPX_TEST_EQ(t1.get_id(), t2.get_id());

            // the migrated object should live on the target now
            HPX_TEST_EQ(t2.call(), target);
            HPX_TEST_EQ(t2.lazy_get_data(), 42);
            HPX_TEST_EQ(t2.lazy_get_base_data(), 7);

            hpx::cout << "." << std::flush;

            std::swap(source, target);
        }

        hpx::cout << std::endl;
    });

    // Second, we generate tons of work which should automatically follow
    // the migration.
    hpx::future<void> create_work = hpx::async([t1, N]() {
        for (std::size_t i = 0; i < 2 * N; ++i)
        {
            hpx::cout << hpx::naming::get_locality_id_from_id(t1.call())
                      << std::flush;
            HPX_TEST_EQ(t1.lazy_get_data(), 42);
            HPX_TEST_EQ(t1.lazy_get_base_data(), 7);
        }
    });

    hpx::wait_all(migrate_future, create_work);

    // rethrow exceptions
    try
    {
        migrate_future.get();
        create_work.get();
    }
    catch (hpx::exception const& e)
    {
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
        hpx::cout << "test migrate polymorphic component: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_polymorphic_component(hpx::find_here(), id));
        hpx::cout << "test migrate polymorphic component: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_polymorphic_component(id, hpx::find_here()));

        hpx::cout << "test migrate lazy polymorphic component: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_polymorphic_component(hpx::find_here(), id));
        hpx::cout << "test migrate lazy polymorphic component: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_polymorphic_component(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test migrate busy polymorphic component: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_busy_polymorphic_component(hpx::find_here(), id));
        hpx::cout << "test migrate busy polymorphic component: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_busy_polymorphic_component(id, hpx::find_here()));

        hpx::cout << "test migrate lazy busy polymorphic component: ->" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_busy_polymorphic_component(hpx::find_here(), id));
        hpx::cout << "test migrate lazy busy polymorphic component: <-" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_busy_polymorphic_component(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test migrate polymorphic component2: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_polymorphic_component2(hpx::find_here(), id));
        hpx::cout << "test migrate polymorphic component2: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_polymorphic_component2(id, hpx::find_here()));

        hpx::cout << "test migrate lazy polymorphic component2: ->" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_polymorphic_component2(hpx::find_here(), id));
        hpx::cout << "test migrate lazy polymorphic component2: <-" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_lazy_polymorphic_component2(id, hpx::find_here()));
    }

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test migrate busy polymorphic component2: ->" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_busy_polymorphic_component2(hpx::find_here(), id));
        hpx::cout << "test migrate busy polymorphic component2: <-" << id
                  << std::endl;
        HPX_TEST(
            test_migrate_busy_polymorphic_component2(id, hpx::find_here()));

        hpx::cout << "test migrate lazy busy polymorphic component2: ->" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_busy_polymorphic_component2(
            hpx::find_here(), id));
        hpx::cout << "test migrate lazy busy polymorphic component2: <-" << id
                  << std::endl;
        HPX_TEST(test_migrate_lazy_busy_polymorphic_component2(
            id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}
