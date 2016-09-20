//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/compression_registration.hpp>
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
        std::true_type(), std::move(dest), std::move(addr),
        hpx::actions::typed_continuation<hpx::id_type>(cont),
        Action(), hpx::threads::thread_priority_normal,
        std::forward<T>(data));

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

#if defined(HPX_HAVE_COMPRESSION_BZIP2)
HPX_ACTION_USES_BZIP2_COMPRESSION(test1_action)
#elif defined(HPX_HAVE_COMPRESSION_ZLIB)
HPX_ACTION_USES_ZLIB_COMPRESSION(test1_action)
#elif defined(HPX_HAVE_COMPRESSION_SNAPPY)
HPX_ACTION_USES_SNAPPY_COMPRESSION(test1_action)
#endif

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
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p;
        auto f = p.get_future();

        parcels.push_back(
            generate_parcel<test1_action>(c.get_id(), p.get_id(), data)
        );

        results.push_back(std::move(f));
    }

    // send parcels
    hpx::get_runtime().get_parcel_handler().put_parcels(std::move(parcels));

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

#if defined(HPX_HAVE_COMPRESSION_BZIP2)
HPX_ACTION_USES_BZIP2_COMPRESSION(test2_action)
#elif defined(HPX_HAVE_COMPRESSION_ZLIB)
HPX_ACTION_USES_ZLIB_COMPRESSION(test2_action)
#elif defined(HPX_HAVE_COMPRESSION_SNAPPY)
HPX_ACTION_USES_SNAPPY_COMPRESSION(test2_action)
#endif

HPX_PLAIN_ACTION(test2, test2_action);

void test_future_argument(hpx::id_type const& id)
{
    std::vector<hpx::lcos::local::promise<double> > args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::local::promise<double> p_arg;
        hpx::lcos::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        parcels.push_back(
            generate_parcel<test2_action>(id, p_cont.get_id(),
                p_arg.get_future())
        );

        args.push_back(std::move(p_arg));
        results.push_back(std::move(f_cont));
    }

    // send parcels
    hpx::get_runtime().get_parcel_handler().put_parcels(std::move(parcels));

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
    std::vector<hpx::parcelset::parcel> parcels;
    for (std::size_t i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p_cont;
        auto f_cont = p_cont.get_future();

        if (std::rand() % 2)
        {
            parcels.push_back(
                generate_parcel<test1_action>(c.get_id(), p_cont.get_id(), data)
            );
        }
        else
        {
            hpx::lcos::local::promise<double> p_arg;

            parcels.push_back(
                generate_parcel<test2_action>(id, p_cont.get_id(),
                    p_arg.get_future())
            );

            args.push_back(std::move(p_arg));
        }

        results.push_back(std::move(f_cont));
    }

    // send parcels
    hpx::get_runtime().get_parcel_handler().put_parcels(std::move(parcels));

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

        std::string serialize_name = serialize_counter.get_name(hpx::launch::sync);
        std::string data_name = data_counter.get_name(hpx::launch::sync);

        if (data_val != 0 && serialize_val != 0)
        {
            // compression should reduce the transmitted amount of data
            HPX_TEST(data_val >= serialize_val);
        }

        hpx::cout
            << "counter: " << serialize_name
            << ", value: " << serialize_value.get_value<double>()
            << std::endl;
        hpx::cout
            << "counter: " << data_name
            << ", value: " << data_value.get_value<double>()
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

    // make sure compression was actually invoked
    verify_counters();

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

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
