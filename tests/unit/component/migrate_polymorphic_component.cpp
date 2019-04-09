//  Copyright (c) 2019 Maximilian Bremer
//  Copyright (c) 2019 Hartmut Kaiser
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

    int get_base_data() const
    {
        return base_data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(test_server_base, get_base_data,
        get_base_data_action);

    virtual int get_data() const
    {
        HPX_TEST(pin_count() != 0);
        return base_data_;
    }
    int get_data_nonvirt() const { return get_data(); }
    HPX_DEFINE_COMPONENT_ACTION(test_server_base, get_data_nonvirt,
        get_data_action);

    // Components which should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // component's constructor.
    test_server_base(test_server_base && rhs)
      : base_data_(std::move(rhs.base_data_))
    {}

    test_server_base& operator=(test_server_base && rhs)
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

typedef test_server_base::get_base_data_action get_base_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_base_data_action);
HPX_REGISTER_ACTION(get_base_data_action);

typedef test_server_base::get_data_action get_data_action;
HPX_REGISTER_ACTION_DECLARATION(get_data_action);
HPX_REGISTER_ACTION(get_data_action);

///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::abstract_migration_support<
        hpx::components::component_base<test_server>,
        test_server_base>
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
        HPX_TEST(pin_count() != 0);
        return data_;
    }

    // Components that should be migrated using hpx::migrate<> need to
    // be Serializable and CopyConstructable. Components can be
    // MoveConstructable in which case the serialized data is moved into the
    // component's constructor.
    test_server(test_server && rhs)
      : base_type(std::move(rhs)), data_(rhs.data_)
    {}

    test_server& operator=(test_server && rhs)
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
struct test_client
  : hpx::components::client_base<test_client, test_server_base>
{
    using base_type =
        hpx::components::client_base<test_client, test_server_base>;

    test_client() = default;
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}
    test_client(hpx::id_type && id) : base_type(std::move(id)) {}

    hpx::id_type call() const
    {
        return call_action()(this->get_id());
    }

    int get_data() const
    {
        return get_data_action()(this->get_id());
    }

    int get_base_data() const
    {
        return get_base_data_action()(this->get_id());
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
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_remote_localities();

    for (hpx::id_type const& id : localities)
    {
        hpx::cout << "test migrate polymorphic component: ->" << id << std::endl;
        HPX_TEST(test_migrate_polymorphic_component(hpx::find_here(), id));
        hpx::cout << "test migrate polymorphic component: <-" << id << std::endl;
        HPX_TEST(test_migrate_polymorphic_component(id, hpx::find_here()));
    }

    return hpx::util::report_errors();
}

