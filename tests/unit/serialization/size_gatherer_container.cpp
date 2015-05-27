//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/parcelset.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/util/lightweight_test.hpp>

// This function will never be called
int test_function(hpx::serialization::serialize_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function, test_action)

///////////////////////////////////////////////////////////////////////////////
boost::uint32_t get_archive_flags(bool endianess, bool array_opt, bool zerocopy)
{
    // compose archive flags
    boost::uint32_t  out_archive_flags = 0U;
    if (endianess)
        out_archive_flags |= hpx::serialization::endian_little;
    else
        out_archive_flags |= hpx::serialization::endian_big;

    if (!array_opt)
    {
        out_archive_flags |= hpx::serialization::disable_array_optimization;
        out_archive_flags |= hpx::serialization::disable_data_chunking;
    }
    else if (!zerocopy)
    {
        out_archive_flags |= hpx::serialization::disable_data_chunking;
    }

    return out_archive_flags;
}

///////////////////////////////////////////////////////////////////////////////
void test_gather_size(std::size_t data_size, boost::uint32_t out_archive_flags,
    bool continuation)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<boost::uint64_t>(&test_function));

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
    if (!(out_archive_flags | hpx::serialization::disable_data_chunking))
        chunks = new std::vector<hpx::serialization::serialization_chunk>();

    boost::uint32_t dest_locality_id = outp.get_destination_locality_id();

    {
        hpx::serialization::detail::size_gatherer_container gather_size;
        std::size_t gathered_size = 0;

        {
            // gather the required size for the archive
            hpx::serialization::output_archive archive(
                gather_size, out_archive_flags, dest_locality_id, chunks);
            archive << outp;
            gathered_size = archive.bytes_written();
        }

        if (chunks)
            chunks->clear();

        std::vector<char> out_buffer;
        std::size_t written_size = 0;

        out_buffer.resize(gathered_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

        {
            // create an output archive and serialize the parcel
            hpx::serialization::output_archive archive(
                out_buffer, out_archive_flags, dest_locality_id, chunks);
            archive << outp;
            written_size = archive.bytes_written();
        }

        HPX_TEST_LTE(written_size, gathered_size);

        hpx::parcelset::parcel inp;

        {
            // create an input archive and deserialize the parcel
            hpx::serialization::input_archive archive(
                out_buffer, written_size, chunks);

            archive >> inp;
        }

        if (chunks)
            chunks->clear();
    }
}

void test_gather_size(std::size_t data_size, bool cont)
{
    test_gather_size(data_size, get_archive_flags(false, false, false), cont);
    test_gather_size(data_size, get_archive_flags(true,  false, false), cont);
    test_gather_size(data_size, get_archive_flags(false, true,  false), cont);
    test_gather_size(data_size, get_archive_flags(true,  true,  false), cont);
    test_gather_size(data_size, get_archive_flags(false, true,  true),  cont);
    test_gather_size(data_size, get_archive_flags(true,  true,  true),  cont);
}

void test_gather_size(std::size_t data_size)
{
    test_gather_size(data_size, false);
    test_gather_size(data_size,  true);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    for (std::size_t i = 0; i != 22; ++i)
        test_gather_size(1 << i);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

