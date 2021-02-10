//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/config/endian.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/serialization/detail/preprocess_container.hpp>
#include <hpx/util/from_string.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// This function will never be called
int test_function(hpx::serialization::serialize_buffer<double> const&)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function, test_action)

std::size_t get_archive_size(hpx::parcelset::parcel const& p,
    std::uint32_t flags,
    std::vector<hpx::serialization::serialization_chunk>* chunks)
{
    // gather the required size for the archive
    hpx::serialization::detail::preprocess_container gather_size;
    hpx::serialization::output_archive archive(gather_size, flags, chunks);
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
        reinterpret_cast<std::uint64_t>(&test_function));

    // compose archive flags
    std::string endian_out = hpx::get_config_entry("hpx.parcel.endian_out",
        hpx::endian::native == hpx::endian::big ? "big" : "little");

    unsigned int out_archive_flags = 0U;
    if (endian_out == "little")
        out_archive_flags |= hpx::serialization::endian_little;
    else if (endian_out == "big")
        out_archive_flags |= hpx::serialization::endian_big;
    else
    {
        HPX_TEST(endian_out == "little" || endian_out == "big");
    }

    std::string array_optimization =
        hpx::get_config_entry("hpx.parcel.array_optimization", "1");

    if (hpx::util::from_string<int>(array_optimization) == 0)
    {
        out_archive_flags |= hpx::serialization::disable_array_optimization;
        out_archive_flags |= hpx::serialization::disable_data_chunking;
    }
    else
    {
        std::string zero_copy_optimization =
            hpx::get_config_entry("hpx.parcel.zero_copy_optimization", "1");
        if (!zerocopy ||
            hpx::util::from_string<int>(zero_copy_optimization) == 0)
        {
            out_archive_flags |= hpx::serialization::disable_data_chunking;
        }
    }

    // create argument for action
    std::vector<double> data;
    data.resize(data_size);

    hpx::serialization::serialize_buffer<double> buffer(data.data(),
        data.size(), hpx::serialization::serialize_buffer<double>::reference);

    // create a parcel with/without continuation
    hpx::parcelset::parcel outp;
    hpx::naming::gid_type dest = here.get_gid();
    if (continuation)
    {
        outp = hpx::parcelset::parcel(
            hpx::parcelset::detail::create_parcel::call(std::move(dest),
                std::move(addr), hpx::actions::typed_continuation<int>(here),
                test_action(), hpx::threads::thread_priority::normal, buffer));
    }
    else
    {
        outp =
            hpx::parcelset::parcel(hpx::parcelset::detail::create_parcel::call(
                std::move(dest), std::move(addr), test_action(),
                hpx::threads::thread_priority::normal, buffer));
    }

    outp.set_source_id(here);

    std::vector<hpx::serialization::serialization_chunk>* chunks = nullptr;
    if (zerocopy)
        chunks = new std::vector<hpx::serialization::serialization_chunk>();

    //std::uint32_t dest_locality_id = outp.destination_locality_id();
    hpx::chrono::high_resolution_timer t;

    for (std::size_t i = 0; i != iterations; ++i)
    {
        std::size_t arg_size =
            get_archive_size(outp, out_archive_flags, chunks);
        std::vector<char> out_buffer;

        out_buffer.resize(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

        {
            // create an output archive and serialize the parcel
            hpx::serialization::output_archive archive(
                out_buffer, out_archive_flags, chunks);
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

int hpx_main(hpx::program_options::variables_map& vm)
{
    bool print_header = vm.count("no-header") == 0;
    bool continuation = vm.count("continuation") != 0;
    bool zerocopy = vm.count("zerocopy") != 0;

    std::vector<hpx::future<double>> timings;
    for (std::size_t i = 0; i != concurrency; ++i)
    {
        timings.push_back(hpx::async(&benchmark_serialization, data_size,
            iterations, continuation, zerocopy));
    }

    double overall_time = 0;
    for (std::size_t i = 0; i != concurrency; ++i)
        overall_time += timings[i].get();

    if (print_header)
        hpx::cout << "datasize,testcount,average_time[s]\n" << hpx::flush;

    hpx::util::format_to(hpx::cout, "{},{},{}\n", data_size, iterations,
        overall_time / concurrency)
        << hpx::flush;
    hpx::util::print_cdash_timing("Serialization", overall_time / concurrency);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    hpx::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()
        ("concurrency",
            hpx::program_options::value<std::size_t>(&concurrency)
                ->default_value(1),
            "number of concurrent serialization operations (default: 1)")
        ("data_size",
            hpx::program_options::value<std::size_t>(&data_size)
                ->default_value(1),
            "size of data buffer to serialize in bytes (default: 1)")
        ("iterations",
            hpx::program_options::value<std::size_t>(&iterations)
                ->default_value(1000),
            "number of iterations while measuring serialization overhead "
            "(default: 1000)")
        ("continuation", "add a continuation to each created parcel")
        ("zerocopy",
            "use zero copy serialization of bitwise copyable "
            "arguments")
        ("no-header", "do not print out the csv header row")
        ;
    // clang-format on

    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
#endif
