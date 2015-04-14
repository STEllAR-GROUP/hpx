//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>

#include <boost/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::serialization::serialize_buffer<char> buffer_plain_type;

buffer_plain_type bounce_plain(buffer_plain_type const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_ACTION(bounce_plain);

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    buffer_plain_type, serialization_buffer_char);
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    buffer_plain_type, serialization_buffer_char);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::serialization::serialize_buffer<char, std::allocator<char> >
    buffer_allocator_type;

buffer_allocator_type bounce_allocator(buffer_allocator_type const& receive_buffer)
{
    return receive_buffer;
}
HPX_PLAIN_ACTION(bounce_allocator);

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    buffer_allocator_type, serialization_buffer_char_allocator);
HPX_REGISTER_BASE_LCO_WITH_VALUE(
    buffer_allocator_type, serialization_buffer_char_allocator);

///////////////////////////////////////////////////////////////////////////////
template <typename Buffer, typename Action>
void test(hpx::id_type dest, char* send_buffer, std::size_t size)
{
    typedef Buffer buffer_type;
    buffer_type recv_buffer;

    std::vector<hpx::future<buffer_type> > recv_buffers;
    recv_buffers.reserve(10);

    Action act;
    for(std::size_t j = 0; j != 10; ++j)
    {
        recv_buffers.push_back(hpx::async(act, dest,
            buffer_type(send_buffer, size, buffer_type::reference)));
    }
    hpx::wait_all(recv_buffers);

    for (hpx::future<buffer_type>& f : recv_buffers)
    {
        buffer_type b = f.get();
        HPX_TEST_EQ(b.size(), size);
        HPX_TEST(0 == memcmp(b.data(), send_buffer, size));
    }
}

template <typename Allocator>
void test_stateful_allocator(hpx::id_type dest, char* send_buffer,
    std::size_t size, Allocator const& alloc)
{
    typedef buffer_allocator_type buffer_type;
    buffer_type recv_buffer;

    std::vector<hpx::future<buffer_type> > recv_buffers;
    recv_buffers.reserve(10);

    bounce_allocator_action act;
    for(std::size_t j = 0; j != 10; ++j)
    {
        recv_buffers.push_back(hpx::async(act, dest,
            buffer_type(send_buffer, size, buffer_type::reference, alloc)));
    }
    hpx::wait_all(recv_buffers);

    for (hpx::future<buffer_type>& f : recv_buffers)
    {
        buffer_type b = f.get();
        HPX_TEST_EQ(b.size(), size);
        HPX_TEST(0 == memcmp(b.data(), send_buffer, size));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    std::size_t const max_size = 1 << 22;
    std::unique_ptr<char[]> send_buffer(new char[max_size]);

    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        for (std::size_t size = 1; size <= max_size; size *= 2)
        {
            test<buffer_plain_type, bounce_plain_action>(
                loc, send_buffer.get(), size);
            test<buffer_allocator_type, bounce_allocator_action>(
                loc, send_buffer.get(), size);
            test_stateful_allocator(
                loc, send_buffer.get(), size, std::allocator<char>());
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return 0;
}
