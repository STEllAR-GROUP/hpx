//  Copyright (c) 2014 John A. Biddiscombe
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/serialization/serialize_buffer.hpp>
#include <hpx/modules/testing.hpp>

#if !defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
#include <boost/shared_array.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

////----------------------------------------------------------------------------
#define MEMORY_BLOCK_SIZE 0x01000000

////----------------------------------------------------------------------------
template <typename T>
class test_allocator : public std::allocator<T>
{
public:
    typedef T        value_type;
    typedef T*       pointer;
    typedef const T* const_pointer;
    typedef T&       reference;
    typedef const T& const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // we want to make sure anything else uses the std allocator
    template <typename U> struct rebind { typedef std::allocator<U> other; };

    pointer allocate(size_type n, const void* = nullptr)
    {
        HPX_TEST_EQ(n, static_cast<size_type>(MEMORY_BLOCK_SIZE));
        return std::allocator<T>::allocate(n);
    }

    void deallocate(pointer p, size_type n)
    {
        HPX_TEST_EQ(n, static_cast<size_type>(MEMORY_BLOCK_SIZE));
        return std::allocator<T>::deallocate(p, n);
    }

    test_allocator() noexcept: std::allocator<T>() {}
    test_allocator(const test_allocator &a) noexcept: std::allocator<T>(a) {}
    ~test_allocator() noexcept {}
};

//----------------------------------------------------------------------------
typedef hpx::serialization::serialize_buffer<char, test_allocator<char> >
    buffer_allocator_type;

buffer_allocator_type allocator_message(buffer_allocator_type const& receive_buffer)
{
    HPX_TEST_EQ(receive_buffer.size(), static_cast<std::size_t>(MEMORY_BLOCK_SIZE));
    return receive_buffer;
}
HPX_PLAIN_ACTION(allocator_message);
HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(buffer_allocator_type,
    serialization_buffer_char_allocator);
HPX_REGISTER_BASE_LCO_WITH_VALUE(buffer_allocator_type,
    serialization_buffer_char_allocator);

//----------------------------------------------------------------------------
void receive(hpx::naming::id_type dest, char* send_buffer,
        std::size_t size, std::size_t window_size)
{
    std::vector<hpx::future<buffer_allocator_type> > recv_buffers;
    recv_buffers.reserve(window_size);

    allocator_message_action msg;
    for(std::size_t j = 0; j != window_size; ++j)
    {
        recv_buffers.push_back(hpx::async(msg, dest,
            buffer_allocator_type(send_buffer, size,
                buffer_allocator_type::reference)));
    }
    hpx::wait_all(recv_buffers);
}
//----------------------------------------------------------------------------

int hpx_main()
{
    // alloc buffer to send
#if defined(HPX_HAVE_CXX17_SHARED_PTR_ARRAY)
    std::shared_ptr<char[]> send_buffer(new char[MEMORY_BLOCK_SIZE]);
#else
    boost::shared_array<char> send_buffer(new char[MEMORY_BLOCK_SIZE]);
#endif

    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        receive(loc, send_buffer.get(), MEMORY_BLOCK_SIZE, 1);
    }

    return hpx::finalize();
}
//----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
