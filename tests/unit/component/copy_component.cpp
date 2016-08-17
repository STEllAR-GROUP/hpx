//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_scan.hpp>

#include <boost/format.hpp>
#include <boost/ref.hpp>

#include <list>
#include <mutex>
#include <set>

#include <utility>
#include <vector>
#include <chrono>
#include <thread>

HPX_REGISTER_PARTITIONED_VECTOR(int);
///////////////////////////////////////////////////////////////////////////////
struct test_server
  : hpx::components::simple_component_base<test_server>
{
    typedef hpx::components::simple_component_base<test_server> base_type;

    test_server() {}
    ~test_server() {}

    hpx::id_type call() const
    {
        return hpx::find_here();
    }

    // components which should be copied using hpx::copy<> need to
    // be Serializable and CopyConstructable. In the remote case
    // it can be MoveConstructable in which case the serialized data
    // is moved into the components constructor.
    test_server(test_server const& rhs)
      : base_type(rhs)
    {}

    test_server(test_server && rhs)
      : base_type(std::move(rhs))
    {}

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
HPX_REGISTER_ACTION(call_action);

struct test_client
  : hpx::components::client_base<test_client, test_server>
{
    typedef hpx::components::client_base<test_client, test_server>
        base_type;

    test_client() {}
    test_client(hpx::shared_future<hpx::id_type> const& id) : base_type(id) {}

    hpx::id_type call() const { return call_action()(this->get_id()); }
};

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 on same locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life on id
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_here(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(id);
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 here
        test_client t2(hpx::components::copy<test_server>(
            t1.get_id(), hpx::find_here()));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life here
        HPX_TEST_EQ(t2.call(), hpx::find_here());

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////
bool test_copy_component_there(hpx::id_type id)
{
    // create component on given locality
    test_client t1 = test_client::create(hpx::find_here());
    HPX_TEST_NEQ(hpx::naming::invalid_id, t1.get_id());

    try {
        // create a copy of t1 on given locality
        test_client t2(hpx::components::copy<test_server>(t1.get_id(), id));
        HPX_TEST_NEQ(hpx::naming::invalid_id, t2.get_id());

        // the new object should life there
        HPX_TEST_EQ(t2.call(), id);

        return true;
    }
    catch (hpx::exception const&) {
        HPX_TEST(false);
    }

    return false;
}

struct opt
{
    int operator()(int v1, int v2) const
    {
        return v1 + v2;
    }
};


int main()
{
    HPX_TEST(test_copy_component(hpx::find_here()));
    HPX_TEST(test_copy_component_here(hpx::find_here()));
    HPX_TEST(test_copy_component_there(hpx::find_here()));

    std::vector<hpx::id_type> localities = hpx::find_remote_localities();
    for (hpx::id_type const& id : localities)
    {
        HPX_TEST(test_copy_component(id));
        HPX_TEST(test_copy_component_here(id));
        HPX_TEST(test_copy_component_there(id));
    }

/* REMOVE LATER */
    auto policy = hpx::parallel::par(hpx::parallel::task);
    auto op =
        [](double v1, double v2) {
            return v1 + v2;
        };
    int init = 0;

    hpx::partitioned_vector<int> v1(100, hpx::container_layout(hpx::find_all_localities()));
    std::iota(v1.begin(), v1.end(), 1);

    hpx::partitioned_vector<int> v2(100, hpx::container_layout(hpx::find_all_localities()));

    auto first = v1.begin();
    auto last = v1.end();
    auto dest = v2.begin();

    auto res = hpx::parallel::inclusive_scan(policy, first, last, dest, init);

    auto res1 = res.get();

    std::vector<int> e(v1.size());
    std::iota(e.begin(), e.end(), 1);
    std::vector<int> f(e.size());

    auto res2 = hpx::parallel::v1::detail::sequential_inclusive_scan(
        e.begin(), e.end(), f.begin(), 0, opt());

    for (auto i : v2) {
        std::cout << i << "\t";
    }

    std::cout << std::endl;

    for (auto i : f) {
        std::cout << i << "\t";
    }

    std::cout << std::endl;

    std::string test = "true";
    for (std::size_t i = 0; i < v1.size(); ++i) {
        if (v2[i] != f[i]) {
            test = "false";
        }
    }

    std::cout << "result: " << test << std::endl;
    std::cout << "end: " << *(res1-1) << " " << *(res2-1) << std::endl;
/* REMOVE LATER END */

    return hpx::util::report_errors();
}

