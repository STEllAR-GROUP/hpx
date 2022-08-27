//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE) && defined(HPX_HAVE_COMPRESSION_BZIP2)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/compression_bzip2.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::size_t const vsize_default = 1024;
std::size_t const numparcels_default = 10;

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
hpx::parcelset::parcel generate_parcel(
    hpx::id_type const& dest_id, hpx::id_type const& cont, T&& data)
{
    hpx::naming::address addr;
    hpx::naming::gid_type dest = dest_id.get_gid();
    hpx::naming::detail::strip_credits_from_gid(dest);
    hpx::parcelset::parcel p(hpx::parcelset::detail::create_parcel::call(
        std::move(dest), std::move(addr),
        hpx::actions::typed_continuation<hpx::id_type>(cont), Action(),
        hpx::threads::thread_priority::normal, std::forward<T>(data)));

    p.set_source_id(hpx::find_here());
    p.size() = 4096;

    return p;
}

///////////////////////////////////////////////////////////////////////////////
struct test_server : hpx::components::component_base<test_server>
{
    hpx::id_type test1(std::vector<double> const& data)
    {
        return hpx::find_here();
    }

    HPX_DEFINE_COMPONENT_ACTION(test_server, test1, test1_action)
};

typedef hpx::components::component<test_server> server_type;
HPX_REGISTER_COMPONENT(server_type, test_server)

typedef test_server::test1_action test1_action;

HPX_REGISTER_ACTION_DECLARATION(test1_action)
HPX_ACTION_USES_BZIP2_COMPRESSION(test1_action)
HPX_REGISTER_ACTION(test1_action)

///////////////////////////////////////////////////////////////////////////////
void test_plain_argument(hpx::id_type const& id)
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<hpx::future<hpx::id_type>> results;
    results.reserve(numparcels_default);

    hpx::components::client<test_server> c = hpx::new_<test_server>(id);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::distributed::promise<hpx::id_type> p;
        auto f = p.get_future();

        parcels.push_back(
            generate_parcel<test1_action>(c.get_id(), p.get_id(), data));

        results.push_back(std::move(f));
    }

    // send parcels
    hpx::get_runtime_distributed().get_parcel_handler().put_parcels(
        std::move(parcels));

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST_EQ(f.get(), id);
    }
}

///////////////////////////////////////////////////////////////////////////////
hpx::id_type test2(hpx::future<double> const& data)
{
    return hpx::find_here();
}

HPX_DECLARE_PLAIN_ACTION(test2, test2_action);
HPX_ACTION_USES_BZIP2_COMPRESSION(test2_action)
HPX_PLAIN_ACTION(test2, test2_action)

void test_future_argument(hpx::id_type const& id)
{
    std::vector<hpx::promise<double>> args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type>> results;
    results.reserve(numparcels_default);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::promise<double> p_arg;
        hpx::distributed::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        parcels.push_back(generate_parcel<test2_action>(
            id, p_cont.get_id(), p_arg.get_future()));

        args.push_back(std::move(p_arg));
        results.push_back(std::move(f_cont));
    }

    // send parcels
    hpx::get_runtime_distributed().get_parcel_handler().put_parcels(
        std::move(parcels));

    // now make the futures ready
    for (hpx::promise<double>& arg : args)
    {
        arg.set_value(42.0);
    }

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST_EQ(f.get(), id);
    }
}

void test_mixed_arguments(hpx::id_type const& id)
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<hpx::promise<double>> args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type>> results;
    results.reserve(numparcels_default);

    hpx::components::client<test_server> c = hpx::new_<test_server>(id);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::distributed::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        if (std::rand() % 2)
        {
            parcels.push_back(generate_parcel<test1_action>(
                c.get_id(), p_cont.get_id(), data));
        }
        else
        {
            hpx::promise<double> p_arg;

            parcels.push_back(generate_parcel<test2_action>(
                id, p_cont.get_id(), p_arg.get_future()));

            args.push_back(std::move(p_arg));
        }

        results.push_back(std::move(f_cont));
    }

    // send parcels
    hpx::get_runtime_distributed().get_parcel_handler().put_parcels(
        std::move(parcels));

    // now make the futures ready
    for (hpx::promise<double>& arg : args)
    {
        arg.set_value(42.0);
    }

    // verify all messages got actually sent to the correct locality
    hpx::wait_all(results);

    for (hpx::future<hpx::id_type>& f : results)
    {
        HPX_TEST_EQ(f.get(), id);
    }
}

///////////////////////////////////////////////////////////////////////////////
void verify_counters()
{
    using namespace hpx::performance_counters;

    std::vector<performance_counter> data_counters =
        discover_counters("/data/count/*/*");
    std::vector<performance_counter> serialize_counters =
        discover_counters("/serialize/count/*/*");

    HPX_TEST_EQ(data_counters.size(), serialize_counters.size());

    for (std::size_t i = 0; i != data_counters.size(); ++i)
    {
        performance_counter const& serialize_counter = serialize_counters[i];
        performance_counter const& data_counter = data_counters[i];

        counter_value serialize_value =
            serialize_counter.get_counter_value(hpx::launch::sync);
        counter_value data_value =
            data_counter.get_counter_value(hpx::launch::sync);

        double serialize_val = serialize_value.get_value<double>();
        double data_val = data_value.get_value<double>();

        std::string serialize_name =
            serialize_counter.get_name(hpx::launch::sync);
        std::string data_name = data_counter.get_name(hpx::launch::sync);

        if (data_val != 0 && serialize_val != 0)
        {
            // compression should reduce the transmitted amount of data
            HPX_TEST_LTE(serialize_val, data_val);
        }

        std::cout << "counter: " << serialize_name
                  << ", value: " << serialize_value.get_value<double>()
                  << std::endl;
        std::cout << "counter: " << data_name
                  << ", value: " << data_value.get_value<double>() << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
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

    // make sure compression was actually invoked
    verify_counters();

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

#endif
