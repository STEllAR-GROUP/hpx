//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/parcel_coalescing.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::size_t const vsize_default = 1024;
std::size_t const numparcels_default = 10;

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
hpx::parcelset::parcel
generate_parcel(hpx::id_type const& dest_id, hpx::id_type const& cont, T && data)
{
    hpx::naming::address addr;
    hpx::naming::gid_type dest = dest_id.get_gid();
    hpx::naming::detail::strip_credits_from_gid(dest);
    hpx::parcelset::parcel p(hpx::parcelset::detail::create_parcel::call(
        std::true_type(), std::true_type(),
        std::move(dest), std::move(addr),
        hpx::actions::typed_continuation<hpx::id_type>(cont),
        Action(), hpx::threads::thread_priority_normal,
        std::forward<T>(data)));

    p.set_source_id(hpx::find_here());
    p.size() = 4096;

    return p;
}

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    typedef hpx::components::component_base<test_server> base_type;

    hpx::id_type test1(std::vector<double> const& data)
    {
        return hpx::find_here();
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, test1, test1_action);
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server);

typedef test_server::test1_action test1_action;

HPX_REGISTER_ACTION_DECLARATION(test1_action);
HPX_ACTION_USES_MESSAGE_COALESCING(test1_action);
HPX_REGISTER_ACTION(test1_action);

///////////////////////////////////////////////////////////////////////////////
void test_plain_argument(hpx::id_type const& id)
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    hpx::components::client<test_server> c = hpx::new_<test_server>(id);

    // create parcels
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p;
        auto f = p.get_future();

        // send parcel
        hpx::get_runtime().get_parcel_handler().put_parcel(
            generate_parcel<test1_action>(c.get_id(), p.get_id(), data));

        results.push_back(std::move(f));
    }

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST(f.get() == id);
    }
}

///////////////////////////////////////////////////////////////////////////////
hpx::id_type test2(hpx::future<double> const& data)
{
    return hpx::find_here();
}
HPX_DECLARE_PLAIN_ACTION(test2, test2_action);
HPX_ACTION_USES_MESSAGE_COALESCING(test2_action);
HPX_PLAIN_ACTION(test2, test2_action);

void test_future_argument(hpx::id_type const& id)
{
    std::vector<hpx::lcos::local::promise<double> > args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    // create parcels
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::local::promise<double> p_arg;
        hpx::lcos::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        // send parcel
        hpx::get_runtime().get_parcel_handler().put_parcel(
            generate_parcel<test2_action>(id, p_cont.get_id(),
                p_arg.get_future()));

        args.push_back(std::move(p_arg));
        results.push_back(std::move(f_cont));
    }


    // now make the futures ready
    for (hpx::lcos::local::promise<double>& arg : args)
    {
        arg.set_value(42.0);
    }

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST(f.get() == id);
    }
}

void test_mixed_arguments(hpx::id_type const& id)
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<hpx::lcos::local::promise<double> > args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    hpx::components::client<test_server> c = hpx::new_<test_server>(id);

    // create parcels
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        if (std::rand() % 2)
        {
            // send parcel
            hpx::get_runtime().get_parcel_handler().put_parcel(
                generate_parcel<test1_action>(c.get_id(), p_cont.get_id(), data)
            );
        }
        else
        {
            hpx::lcos::local::promise<double> p_arg;

            // send parcel
            hpx::get_runtime().get_parcel_handler().put_parcel(
                generate_parcel<test2_action>(id, p_cont.get_id(),
                    p_arg.get_future())
            );

            args.push_back(std::move(p_arg));
        }

        results.push_back(std::move(f_cont));
    }

    // now make the futures ready
    for (hpx::lcos::local::promise<double>& arg : args)
    {
        arg.set_value(42.0);
    }

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST(f.get() == id);
    }
}

///////////////////////////////////////////////////////////////////////////////
void print_counters(char const* name)
{
    using namespace hpx::performance_counters;

    std::vector<performance_counter> counters = discover_counters(name);

    for (performance_counter const& c : counters)
    {
        counter_value value = c.get_counter_value(hpx::launch::sync);
        HPX_TEST_NEQ(value.get_value<double>(), 0.0);

        hpx::cout
            << "counter: " << c.get_name(hpx::launch::sync)
            << ", value: " << value.get_value<double>()
            << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    for (hpx::id_type const& id : hpx::find_remote_localities())
    {
        test_plain_argument(id);
        test_future_argument(id);
        test_mixed_arguments(id);
    }

    // make sure coalescing was actually invoked
    print_counters("/coalescing{locality#0/total}/count/parcels@test1_action");
    print_counters("/coalescing{locality#0/total}/count/parcels@test2_action");
    print_counters("/coalescing{locality#0/total}/count/messages@test1_action");
    print_counters("/coalescing{locality#0/total}/count/messages@test2_action");

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // explicitly enable message handlers (parcel coalescing)
    std::vector<std::string> const cfg = {
        "hpx.parcel.message_handlers=1"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
