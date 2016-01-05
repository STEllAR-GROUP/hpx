//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
std::size_t const vsize_default = 1024;
std::size_t const numparcels_default = 10;

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
hpx::parcelset::parcel
generate_parcel(hpx::id_type const& dest, hpx::id_type const& cont, T && data)
{
    hpx::naming::address addr;
    hpx::parcelset::parcel p(dest, addr,
        hpx::actions::typed_continuation<hpx::id_type>(cont),
        Action(), hpx::threads::thread_priority_normal,
        std::forward<T>(data));

    p.parcel_id() = hpx::parcelset::parcel::generate_unique_id();
    p.set_source_id(hpx::find_here());

    return p;
}

///////////////////////////////////////////////////////////////////////////////
hpx::id_type test1(std::vector<double> const& data)
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(test1);
HPX_ACTION_USES_MESSAGE_COALESCING(test1_action);

void test_plain_argument(hpx::id_type const& id)
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (int i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p;
        parcels.push_back(
            generate_parcel<test1_action>(id, p.get_id(), data)
        );
        results.push_back(p.get_future());
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
HPX_PLAIN_ACTION(test2);
HPX_ACTION_USES_MESSAGE_COALESCING(test2_action);

void test_future_argument(hpx::id_type const& id)
{
    std::vector<hpx::lcos::local::promise<double> > args;
    args.reserve(numparcels_default);

    std::vector<hpx::future<hpx::id_type> > results;
    results.reserve(numparcels_default);

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (int i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::local::promise<double> p_arg;
        hpx::lcos::promise<hpx::id_type> p_cont;

        parcels.push_back(
            generate_parcel<test2_action>(id, p_cont.get_id(),
                p_arg.get_future())
        );

        args.push_back(std::move(p_arg));
        results.push_back(p_cont.get_future());
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

    // create parcels
    std::vector<hpx::parcelset::parcel> parcels;
    for (int i = 0; i != numparcels_default; ++i)
    {
        hpx::lcos::promise<hpx::id_type> p_cont;

        if (std::rand() % 2)
        {
            parcels.push_back(
                generate_parcel<test1_action>(id, p_cont.get_id(), data)
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

        results.push_back(p_cont.get_future());
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
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.parcel.message_handlers=1");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
