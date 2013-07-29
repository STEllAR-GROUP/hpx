//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/serialize_buffer.hpp>

#include <hpx/include/iostreams.hpp>

#include <boost/format.hpp>

// This function will never be called
void test_function(hpx::util::serialize_buffer<double> const& b)
{
}
HPX_PLAIN_ACTION(test_function, test_action)

///////////////////////////////////////////////////////////////////////////////
double benchmark_serialization(std::size_t data_size, std::size_t iterations)
{
    hpx::naming::gid_type here = hpx::find_here().get_gid();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid, reinterpret_cast<boost::uint64_t>(&test_function));

    // compose archive flags
#ifdef BOOST_BIG_ENDIAN
    std::string endian_out =
        hpx::get_config_entry("hpx.parcel.endian_out", "big");
#else
    std::string endian_out =
        hpx::get_config_entry("hpx.parcel.endian_out", "little");
#endif

    int in_archive_flags = boost::archive::no_header;
    int out_archive_flags = boost::archive::no_header;
    if (endian_out == "little")
        out_archive_flags |= hpx::util::endian_little;
    else if (endian_out == "big")
        out_archive_flags |= hpx::util::endian_big;
    else {
        BOOST_ASSERT(endian_out =="little" || endian_out == "big");
    }

    std::string array_optimization =
        hpx::get_config_entry("hpx.parcel.array_optimization", "1");
    if (boost::lexical_cast<int>(array_optimization) == 0)
    {
        in_archive_flags |= hpx::util::disable_array_optimization;
        out_archive_flags |= hpx::util::disable_array_optimization;
    }

    // create argument for action
    std::vector<double> data;
    data.resize(data_size);

    hpx::util::serialize_buffer<double> buffer(data.data(), data.size(),
        hpx::util::serialize_buffer<double>::reference);

    // create a parcel without continuation
    hpx::parcelset::parcel outp (here, addr,
        new hpx::actions::transfer_action<test_action>(
            hpx::threads::thread_priority_normal, buffer));

    hpx::util::high_resolution_timer t;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        std::size_t arg_size = hpx::traits::get_type_size(outp);
        std::vector<char> out_buffer;

        out_buffer.resize(arg_size*2);

        {
            // create an output archive and serialize the parcel
            hpx::util::portable_binary_oarchive archive(
                out_buffer, 0, out_archive_flags);
            archive << outp;

            arg_size = archive.bytes_written();
        }

        hpx::parcelset::parcel inp;

        {
            // create an input archive and deserialize the parcel
            hpx::util::portable_binary_iarchive archive(
                out_buffer, arg_size, in_archive_flags);

            archive >> inp;
        }
    }

    return t.elapsed();
}

///////////////////////////////////////////////////////////////////////////////
std::size_t data_size = 1000000;
std::size_t iterations = 1000;
std::size_t concurrency = 1;

int hpx_main(boost::program_options::variables_map& vm)
{
    bool print_header = (vm.count("no-header") == 0) ? true : false;

    std::vector<hpx::future<double> > timings;
    for (int i = 0; i != concurrency; ++i)
    {
        timings.push_back(hpx::async(
            hpx::util::bind(&benchmark_serialization, data_size, iterations)));
    }

    double overall_time = 0;
    for (int i = 0; i != concurrency; ++i)
        overall_time += timings[i].get();

    if (print_header)
        hpx::cout << "datasize,testcount,time/test[ns]\n" << hpx::flush;

    hpx::cout << (boost::format("%d,%d,%f\n") %
        data_size % iterations % (overall_time / concurrency)) << hpx::flush;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "concurrency"
        , boost::program_options::value<std::size_t>(&concurrency)->default_value(1)
        , "number of concurrent serialization operations (default: 1)")

        ( "data_size"
        , boost::program_options::value<std::size_t>(&data_size)->default_value(1)
        , "size of data buffer to serialize in bytes (default: 1)")

        ( "iterations"
        , boost::program_options::value<std::size_t>(&iterations)->default_value(1000)
        , "number of iterations while measuring serialization overhead (default: 1000)")

        ( "no-header"
        , "do not print out the csv header row")
        ;

    return hpx::init(cmdline, argc, argv);
}

