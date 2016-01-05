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
hpx::id_type test1(std::vector<double> const& data)
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(test1);

hpx::parcelset::parcel
generate_parcel_plain(hpx::id_type const& dest, hpx::id_type const& cont,
    std::vector<double> const& data)
{
    hpx::naming::address addr(dest.get_gid(),
        hpx::components::component_plain_function,
        reinterpret_cast<boost::uint64_t>(&test1));

    hpx::parcelset::parcel p(dest, addr,
        hpx::actions::typed_continuation<hpx::id_type>(cont),
        test1_action(), hpx::threads::thread_priority_normal, data);

    p.parcel_id() = hpx::parcelset::parcel::generate_unique_id();
    p.set_source_id(hpx::find_here());

    return p;
}

void test_plain_argument()
{
    std::vector<double> data(vsize_default);
    std::generate(data.begin(), data.end(), std::rand);

    for (hpx::id_type const& id : hpx::find_remote_localities())
    {
        std::vector<hpx::future<hpx::id_type> > results;
        results.reserve(numparcels_default);

        // create parcels
        std::vector<hpx::parcelset::parcel> parcels;
        for (int i = 0; i != numparcels_default; ++i)
        {
            hpx::lcos::promise<hpx::id_type> p;
            parcels.push_back(generate_parcel_plain(id, p.get_id(), data));
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
}

///////////////////////////////////////////////////////////////////////////////
hpx::id_type test2(hpx::future<double> const& data)
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(test2);

hpx::parcelset::parcel
generate_parcel_future(hpx::id_type const& dest, hpx::id_type const& cont,
    hpx::future<double> && data)
{
    hpx::naming::address addr(dest.get_gid(),
        hpx::components::component_plain_function,
        reinterpret_cast<boost::uint64_t>(&test2));

    hpx::parcelset::parcel p(dest, addr,
        hpx::actions::typed_continuation<hpx::id_type>(cont),
        test2_action(), hpx::threads::thread_priority_normal, std::move(data));

    p.parcel_id() = hpx::parcelset::parcel::generate_unique_id();
    p.set_source_id(hpx::find_here());

    return p;
}

void test_future_argument()
{
    for (hpx::id_type const& id : hpx::find_remote_localities())
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

            parcels.push_back(generate_parcel_future(id, p_cont.get_id(),
                p_arg.get_future()));

            args.push_back(std::move(p_arg));
            results.push_back(p_cont.get_future());
        }

        // send parcels
        hpx::get_runtime().get_parcel_handler().put_parcels(std::move(parcels));

        // now make the future ready
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
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_plain_argument();
    test_future_argument();

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
