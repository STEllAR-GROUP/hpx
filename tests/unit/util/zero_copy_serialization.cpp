//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/serialize_buffer.hpp>
#include <hpx/util/lightweight_test.hpp>

// These functions will never be called
int test_function1(hpx::util::serialize_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function1, test_action1)

int test_function2(hpx::util::serialize_buffer<double> const& b1,
    hpx::util::serialize_buffer<double> const& b2)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function2, test_action2)

///////////////////////////////////////////////////////////////////////////////
void test_parcel_serialization(hpx::parcelset::parcel outp,
    int in_archive_flags, int out_archive_flags)
{
    std::size_t arg_size = hpx::traits::get_type_size(outp);
    std::vector<char> out_buffer;
    std::vector<hpx::util::serialization_chunk> out_chunks;

    out_buffer.resize(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

    {
        // create an output archive and serialize the parcel
        hpx::util::portable_binary_oarchive archive(
            out_buffer, &out_chunks, 0, out_archive_flags);
        archive << outp;

        arg_size = archive.bytes_written();
    }

    hpx::parcelset::parcel inp;

    {
        // create an input archive and deserialize the parcel
        hpx::util::portable_binary_iarchive archive(
            out_buffer, &out_chunks, arg_size, in_archive_flags);

        archive >> inp;
    }

    // make sure the parcel has been deserialized properly
    HPX_TEST_EQ(outp.get_parcel_id(), inp.get_parcel_id());
    HPX_TEST_EQ(outp.get_source(), inp.get_source());
    HPX_TEST_EQ(outp.get_destination_locality(), inp.get_destination_locality());
    HPX_TEST_EQ(outp.get_start_time(), inp.get_start_time());

    hpx::actions::action_type outact = outp.get_action();
    hpx::actions::action_type inact = inp.get_action();

    HPX_TEST_EQ(outact->get_component_type(), inact->get_component_type());
    HPX_TEST_EQ(outact->get_action_name(), inact->get_action_name());
    HPX_TEST_EQ(outact->get_action_type(), inact->get_action_type());
    HPX_TEST_EQ(outact->get_parent_locality_id(), inact->get_parent_locality_id());
    HPX_TEST_EQ(outact->get_parent_thread_id(), inact->get_parent_thread_id());
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());
    HPX_TEST_EQ(outact->get_thread_priority(), inact->get_thread_priority());
    HPX_TEST_EQ(outact->get_thread_stacksize(), inact->get_thread_stacksize());
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());

    hpx::actions::continuation_type outcont = outp.get_continuation();
    hpx::actions::continuation_type incont = inp.get_continuation();

    HPX_TEST_EQ(outcont->get_continuation_name(), incont->get_continuation_name());
    HPX_TEST_EQ(outcont->get_gid(), incont->get_gid());
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void test_normal_serialization(T& arg)
{
    hpx::naming::id_type const here_id = hpx::find_here();
    hpx::naming::gid_type here = here_id.get_gid();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<boost::uint64_t>(&test_function1));

    // compose archive flags
    int in_archive_flags = boost::archive::no_header;
    int out_archive_flags = boost::archive::no_header;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::util::endian_big;
#else
    out_archive_flags |= hpx::util::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::parcelset::parcel outp(here, addr,
        new hpx::actions::transfer_action<test_action1>(
            hpx::threads::thread_priority_normal, arg),
        new hpx::actions::typed_continuation<int>(here_id));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here_id);

    std::size_t arg_size = hpx::traits::get_type_size(outp);
    std::vector<char> out_buffer;

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void test_zero_copy_serialization(T& arg)
{
    hpx::naming::id_type const here_id = hpx::find_here();
    hpx::naming::gid_type here = here_id.get_gid();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<boost::uint64_t>(&test_function1));

    // compose archive flags
    int in_archive_flags = boost::archive::no_header;
    int out_archive_flags = boost::archive::no_header;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::util::endian_big;
#else
    out_archive_flags |= hpx::util::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::parcelset::parcel outp(here, addr,
        new hpx::actions::transfer_action<test_action1>(
            hpx::threads::thread_priority_normal, arg),
        new hpx::actions::typed_continuation<int>(here_id));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here_id);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags);
}

template <typename T1, typename T2>
void test_zero_copy_serialization(T1& arg1, T2& arg2)
{
    hpx::naming::id_type const here_id = hpx::find_here();
    hpx::naming::gid_type here = here_id.get_gid();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<boost::uint64_t>(&test_function2));

    // compose archive flags
    int in_archive_flags = boost::archive::no_header;
    int out_archive_flags = boost::archive::no_header;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::util::endian_big;
#else
    out_archive_flags |= hpx::util::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::parcelset::parcel outp(here, addr,
        new hpx::actions::transfer_action<test_action2>(
            hpx::threads::thread_priority_normal, arg1, arg2),
        new hpx::actions::typed_continuation<int>(here_id));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here_id);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t size = 1;
    for (std::size_t i = 0; i != 20; ++i) {
        // create argument for action
        std::vector<double> data;
        data.resize(size << i);

        hpx::util::serialize_buffer<double> buffer(data.data(), data.size(),
            hpx::util::serialize_buffer<double>::reference);

        test_normal_serialization(buffer);
        test_zero_copy_serialization(buffer);
        test_zero_copy_serialization(buffer, buffer);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

