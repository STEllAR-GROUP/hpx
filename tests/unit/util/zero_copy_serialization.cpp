//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/serialize_buffer.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/foreach.hpp>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct data_buffer
{
    data_buffer() : flag_(false) {}
    data_buffer(std::size_t size) : data_(size, 0), flag_(false) {}

    std::vector<T> data_;
    bool flag_;

    template <typename Archive>
    void save(Archive& ar, unsigned) const
    {
        boost::uint64_t size = data_.size();
        ar & size;
        ar & boost::serialization::make_array(data_.data(), size);
        ar & flag_;
    }

    template <typename Archive>
    void load(Archive& ar, unsigned)
    {
        boost::uint64_t size = 0;
        ar & size;
        data_.resize(size);
        ar & boost::serialization::make_array(data_.data(), size);
        ar & flag_;
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

///////////////////////////////////////////////////////////////////////////////
int test_function1(hpx::util::serialize_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function1, test_action1)

int test_function2(hpx::util::serialize_buffer<double> const& b1,
    hpx::util::serialize_buffer<int> const& b2)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function2, test_action2)

int test_function3(double d,
    hpx::util::serialize_buffer<double> const& b1,
    std::string const& s, int i,
    hpx::util::serialize_buffer<int> const& b2)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function3, test_action3)

int test_function4(data_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function4, test_action4)

///////////////////////////////////////////////////////////////////////////////
void test_parcel_serialization(hpx::parcelset::parcel outp,
    int in_archive_flags, int out_archive_flags, bool zero_copy)
{
    // serialize data
    std::size_t arg_size = hpx::traits::get_type_size(outp);
    std::vector<char> out_buffer;
    std::vector<hpx::util::serialization_chunk> out_chunks;
    boost::uint32_t dest_locality_id = outp.get_destination_locality_id();

    out_buffer.resize(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

    {
        // create an output archive and serialize the parcel
        hpx::util::portable_binary_oarchive archive(
            out_buffer, zero_copy ? &out_chunks : 0, dest_locality_id, 0,
            out_archive_flags);
        archive << outp;

        arg_size = archive.bytes_written();
    }

    out_buffer.resize(arg_size);

    // deserialize data
    hpx::parcelset::parcel inp;

    {
        // create an input archive and deserialize the parcel
        hpx::util::portable_binary_iarchive archive(
            out_buffer, &out_chunks, arg_size, in_archive_flags);

        archive >> inp;
    }

    // make sure the parcel has been de-serialized properly
    HPX_TEST_EQ(outp.get_parcel_id(), inp.get_parcel_id());
    HPX_TEST_EQ(outp.get_source(), inp.get_source());
    HPX_TEST_EQ(outp.get_destination_locality(), inp.get_destination_locality());
    HPX_TEST_EQ(outp.get_start_time(), inp.get_start_time());

    hpx::actions::action_type outact = outp.get_action();
    hpx::actions::action_type inact = inp.get_action();

    HPX_TEST_EQ(outact->get_component_type(), inact->get_component_type());
    HPX_TEST_EQ(outact->get_action_name(), inact->get_action_name());
    HPX_TEST_EQ(int(outact->get_action_type()), int(inact->get_action_type()));
    HPX_TEST_EQ(outact->get_parent_locality_id(), inact->get_parent_locality_id());
    HPX_TEST_EQ(outact->get_parent_thread_id(), inact->get_parent_thread_id());
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());
    HPX_TEST_EQ(int(outact->get_thread_priority()), int(inact->get_thread_priority()));
    HPX_TEST_EQ(int(outact->get_thread_stacksize()), int(inact->get_thread_stacksize()));
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());

    hpx::actions::continuation_type outcont = outp.get_continuation();
    hpx::actions::continuation_type incont = inp.get_continuation();

    HPX_TEST_EQ(outcont->get_continuation_name(), incont->get_continuation_name());
    HPX_TEST_EQ(outcont->get_gid(), incont->get_gid());

    //// invoke action encapsulated in inp
    //naming::address const* inaddrs = pin.get_destination_addrs();
    //hpx::threads::thread_init_data data;
    //inact->get_thread_init_data(inaddrs[0].address_, data);
    //data.func(hpx::threads::wait_signaled);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
void test_normal_serialization(T& arg)
{
    hpx::naming::id_type const here = hpx::find_here();
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
        new hpx::actions::transfer_action<Action>(
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(arg)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, false);
}

template <typename T1, typename T2>
void test_normal_serialization(T1& arg1, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
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
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(arg1, arg2)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, false);
}

template <typename T1, typename T2>
void test_normal_serialization(double d, T1& arg1, std::string const& s,
    int i, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
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
        new hpx::actions::transfer_action<test_action3>(
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(d, arg1, s, i, arg2)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, false);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
void test_zero_copy_serialization(T& arg)
{
    hpx::naming::id_type const here = hpx::find_here();
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
        new hpx::actions::transfer_action<Action>(
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(arg)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, true);
}

template <typename T1, typename T2>
void test_zero_copy_serialization(T1& arg1, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
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
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(arg1, arg2)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, true);
}

template <typename T1, typename T2>
void test_zero_copy_serialization(double d, T1& arg1, std::string const& s,
    int i, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
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
        new hpx::actions::transfer_action<test_action3>(
            hpx::threads::thread_priority_normal,
                hpx::util::forward_as_tuple(d, arg1, s, i, arg2)),
        new hpx::actions::typed_continuation<int>(here));

    outp.set_parcel_id(hpx::parcelset::parcel::generate_unique_id());
    outp.set_source(here);

    test_parcel_serialization(outp, in_archive_flags, out_archive_flags, true);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t size = 1;
    for (std::size_t i = 0; i != 20; ++i) {
        // create argument for action
        std::vector<double> data1;
        data1.resize(size << i);

        hpx::util::serialize_buffer<double> buffer1(data1.data(), data1.size(),
            hpx::util::serialize_buffer<double>::reference);

        test_normal_serialization<test_action1>(buffer1);
        test_zero_copy_serialization<test_action1>(buffer1);

        std::vector<int> data2;
        data2.resize(size << i);

        hpx::util::serialize_buffer<int> buffer2(data2.data(), data2.size(),
            hpx::util::serialize_buffer<int>::reference);

        test_normal_serialization(buffer1, buffer2);
        test_zero_copy_serialization(buffer1, buffer2);
        test_normal_serialization(42.0, buffer1, "42.0", 42, buffer2);
        test_zero_copy_serialization(42.0, buffer1, "42.0", 42, buffer2);

        data_buffer<double> buffer3(size << i);
        test_normal_serialization<test_action4>(buffer3);
        test_zero_copy_serialization<test_action4>(buffer3);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

