//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <algorithm>
#include <iterator>
#include <fstream>

#include <boost/format.hpp>

// This function will never be called
int test_function(hpx::serialization::serialize_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function, test_action)

std::size_t get_archive_size(hpx::parcelset::parcel const& p,
    boost::uint32_t flags,
    std::vector<hpx::serialization::serialization_chunk>* chunks)
{
    // gather the required size for the archive
    hpx::serialization::detail::size_gatherer_container gather_size;
    hpx::serialization::output_archive archive(
        gather_size, flags, 0, chunks);
    archive << p;
    return gather_size.size();
}

///////////////////////////////////////////////////////////////////////////////
double benchmark_serialization(std::size_t data_size, std::size_t iterations,
    bool continuation, bool zerocopy)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<boost::uint64_t>(&test_function));

    // compose archive flags
#ifdef BOOST_BIG_ENDIAN
    std::string endian_out =
        hpx::get_config_entry("hpx.parcel.endian_out", "big");
#else
    std::string endian_out =
        hpx::get_config_entry("hpx.parcel.endian_out", "little");
#endif

    unsigned int out_archive_flags = 0U;
    if (endian_out == "little")
        out_archive_flags |= hpx::serialization::endian_little;
    else if (endian_out == "big")
        out_archive_flags |= hpx::serialization::endian_big;
    else {
        HPX_ASSERT(endian_out =="little" || endian_out == "big");
    }

    std::string array_optimization =
        hpx::get_config_entry("hpx.parcel.array_optimization", "1");

    if (boost::lexical_cast<int>(array_optimization) == 0)
    {
        out_archive_flags |= hpx::serialization::disable_array_optimization;
        out_archive_flags |= hpx::serialization::disable_data_chunking;
    }
    else
    {
        std::string zero_copy_optimization =
            hpx::get_config_entry("hpx.parcel.zero_copy_optimization", "1");
        if (boost::lexical_cast<int>(zero_copy_optimization) == 0)
        {
            out_archive_flags |= hpx::serialization::disable_data_chunking;
        }
    }

    // create argument for action
    std::vector<double> data;
    data.resize(data_size);

    hpx::serialization::serialize_buffer<double> buffer(data.data(), data.size(),
        hpx::serialization::serialize_buffer<double>::reference);

    // create a parcel with/without continuation
    hpx::parcelset::parcel outp;
    if (continuation) {
        outp = hpx::parcelset::parcel(here, addr,
            new hpx::actions::transfer_action<test_action>(
                hpx::threads::thread_priority_normal, buffer),
            new hpx::actions::typed_continuation<int>(here));
    }
    else {
        outp = hpx::parcelset::parcel(here, addr,
            new hpx::actions::transfer_action<test_action>(
                hpx::threads::thread_priority_normal, buffer));
    }

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    std::vector<hpx::serialization::serialization_chunk>* chunks = 0;
    if (zerocopy)
        chunks = new std::vector<hpx::serialization::serialization_chunk>();

    boost::uint32_t dest_locality_id = outp.get_destination_locality_id();
    hpx::util::high_resolution_timer t;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        std::size_t arg_size = get_archive_size(outp, out_archive_flags, chunks);
        std::vector<char> out_buffer;

        out_buffer.resize(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

        {
            // create an output archive and serialize the parcel
            hpx::serialization::output_archive archive(
                out_buffer, out_archive_flags, dest_locality_id, chunks);
            archive << outp;
            arg_size = archive.bytes_written();
        }

        hpx::parcelset::parcel inp;

        {
            // create an input archive and deserialize the parcel
            hpx::serialization::input_archive archive(
                out_buffer, arg_size, chunks);

            archive >> inp;
        }

        if (chunks)
            chunks->clear();
    }

    return t.elapsed();
}

///////////////////////////////////////////////////////////////////////////////
std::size_t data_size = 1;
std::size_t iterations = 1000;
std::size_t concurrency = 1;

int hpx_main(boost::program_options::variables_map& vm)
{
    bool print_header = vm.count("no-header") == 0;
    bool continuation = vm.count("continuation") != 0;
    bool zerocopy = vm.count("zerocopy") != 0;

    std::vector<hpx::future<double> > timings;
    for (std::size_t i = 0; i != concurrency; ++i)
    {
        timings.push_back(hpx::async(
            &benchmark_serialization, data_size, iterations,
            continuation, zerocopy));
    }

    double overall_time = 0;
    for (std::size_t i = 0; i != concurrency; ++i)
        overall_time += timings[i].get();

    if (print_header)
        hpx::cout << "datasize,testcount,average_time[s]\n" << hpx::flush;

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

        ( "continuation"
        , "add a continuation to each created parcel")

        ( "zerocopy"
        , "use zero copy serialization of bitwise copyable arguments")

        ( "no-header"
        , "do not print out the csv header row")
        ;

    return hpx::init(cmdline, argc, argv);
}

