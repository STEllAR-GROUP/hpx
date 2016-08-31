//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/runtime/actions/transfer_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>
#include <hpx/runtime/serialization/detail/preprocess.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

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
        std::uint64_t size = data_.size();
        ar & size;
        ar & hpx::serialization::make_array(data_.data(), size);
        ar & flag_;
    }

    template <typename Archive>
    void load(Archive& ar, unsigned)
    {
        std::uint64_t size = 0;
        ar & size;
        data_.resize(size);
        ar & hpx::serialization::make_array(data_.data(), size);
        ar & flag_;
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()
};

///////////////////////////////////////////////////////////////////////////////
int test_function1(hpx::serialization::serialize_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function1, test_action1)

int test_function2(hpx::serialization::serialize_buffer<double> const& b1,
    hpx::serialization::serialize_buffer<int> const& b2)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function2, test_action2)

int test_function3(double d,
    hpx::serialization::serialize_buffer<double> const& b1,
    std::string const& s, int i,
    hpx::serialization::serialize_buffer<int> const& b2)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function3, test_action3)

int test_function4(data_buffer<double> const& b)
{
    return 42;
}
HPX_PLAIN_ACTION(test_function4, test_action4)

std::size_t get_archive_size(hpx::parcelset::parcel const& p,
    std::uint32_t flags,
    std::vector<hpx::serialization::serialization_chunk>* chunks)
{
    // gather the required size for the archive
    hpx::serialization::detail::preprocess gather_size;
    hpx::serialization::output_archive archive(gather_size, flags, chunks);
    archive << p;
    return gather_size.size();
}

///////////////////////////////////////////////////////////////////////////////
void test_parcel_serialization(hpx::parcelset::parcel outp,
    int out_archive_flags, bool zero_copy)
{
    // serialize data
    std::vector<hpx::serialization::serialization_chunk> out_chunks;
    std::size_t arg_size = get_archive_size(outp, out_archive_flags,
        zero_copy ? &out_chunks : nullptr);
    std::vector<char> out_buffer;
    std::uint32_t dest_locality_id = outp.destination_locality_id();

    out_buffer.resize(arg_size + HPX_PARCEL_SERIALIZATION_OVERHEAD);

    {
        // create an output archive and serialize the parcel
        hpx::serialization::output_archive archive(
            out_buffer, out_archive_flags,
            zero_copy ? &out_chunks : nullptr);
        archive << outp;

        arg_size = archive.bytes_written();
    }

    out_buffer.resize(arg_size);

    // deserialize data
    hpx::parcelset::parcel inp;

    {
        // create an input archive and deserialize the parcel
        hpx::serialization::input_archive archive(
            out_buffer, arg_size, &out_chunks);

        archive >> inp;
    }

    // make sure the parcel has been de-serialized properly
    HPX_TEST_EQ(outp.source_id(), inp.source_id());
    HPX_TEST_EQ(outp.destination_locality(), inp.destination_locality());
    HPX_TEST_EQ(outp.start_time(), inp.start_time());

    hpx::actions::base_action *outact = outp.get_action();
    hpx::actions::base_action *inact = inp.get_action();

    HPX_TEST_EQ(outact->get_component_type(), inact->get_component_type());
    HPX_TEST_EQ(outact->get_action_name(), inact->get_action_name());
    HPX_TEST_EQ(int(outact->get_action_type()), int(inact->get_action_type()));
    HPX_TEST_EQ(outact->get_parent_locality_id(), inact->get_parent_locality_id());
    HPX_TEST_EQ(outact->get_parent_thread_id(), inact->get_parent_thread_id());
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());
    HPX_TEST_EQ(int(outact->get_thread_priority()), int(inact->get_thread_priority()));
    HPX_TEST_EQ(int(outact->get_thread_stacksize()), int(inact->get_thread_stacksize()));
    HPX_TEST_EQ(outact->get_parent_thread_phase(), inact->get_parent_thread_phase());

    std::unique_ptr<hpx::actions::continuation> outcont = outp.get_continuation();
    std::unique_ptr<hpx::actions::continuation> incont = inp.get_continuation();

    HPX_TEST_EQ(outcont->get_continuation_name(), incont->get_continuation_name());
    HPX_TEST_EQ(outcont->get_id(), incont->get_id());

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
        reinterpret_cast<std::uint64_t>(&test_function1));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            Action(), hpx::threads::thread_priority_normal, arg
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, false);
}

template <typename T1, typename T2>
void test_normal_serialization(T1& arg1, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<std::uint64_t>(&test_function2));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            test_action2(), hpx::threads::thread_priority_normal, arg1, arg2
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, false);
}

template <typename T1, typename T2>
void test_normal_serialization(double d, T1& arg1, std::string const& s,
    int i, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<std::uint64_t>(&test_function2));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            test_action3(),
            hpx::threads::thread_priority_normal, d, arg1, s, i, arg2
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, false);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Action, typename T>
void test_zero_copy_serialization(T& arg)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<std::uint64_t>(&test_function1));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            Action(), hpx::threads::thread_priority_normal, arg
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, true);
}

template <typename T1, typename T2>
void test_zero_copy_serialization(T1& arg1, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<std::uint64_t>(&test_function2));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            test_action2(), hpx::threads::thread_priority_normal, arg1, arg2
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, true);
}

template <typename T1, typename T2>
void test_zero_copy_serialization(double d, T1& arg1, std::string const& s,
    int i, T2& arg2)
{
    hpx::naming::id_type const here = hpx::find_here();
    hpx::naming::address addr(hpx::get_locality(),
        hpx::components::component_invalid,
        reinterpret_cast<std::uint64_t>(&test_function2));

    // compose archive flags
    unsigned out_archive_flags = 0U;
#ifdef BOOST_BIG_ENDIAN
    out_archive_flags |= hpx::serialization::endian_big;
#else
    out_archive_flags |= hpx::serialization::endian_little;
#endif

    // create a parcel with/without continuation
    hpx::naming::gid_type dest = here.get_gid();
    hpx::parcelset::parcel outp(
        hpx::parcelset::detail::create_parcel::call(std::true_type(), std::true_type(),
            std::move(dest), std::move(addr),
            hpx::actions::typed_continuation<int>(here),
            test_action3(), hpx::threads::thread_priority_normal, d, arg1, s, i, arg2
        )
    );

    outp.set_source_id(here);

    test_parcel_serialization(std::move(outp), out_archive_flags, true);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t size = 1;
    for (std::size_t i = 0; i != 20; ++i) {
        // create argument for action
        std::vector<double> data1;
        data1.resize(size << i);

        hpx::serialization::serialize_buffer<double> buffer1(data1.data(), data1.size(),
            hpx::serialization::serialize_buffer<double>::reference);

        test_normal_serialization<test_action1>(buffer1);
        test_zero_copy_serialization<test_action1>(buffer1);

        std::vector<int> data2;
        data2.resize(size << i);

        hpx::serialization::serialize_buffer<int> buffer2(data2.data(), data2.size(),
            hpx::serialization::serialize_buffer<int>::reference);

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

