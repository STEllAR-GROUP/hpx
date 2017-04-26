//  Copyright (c) 2014 Jeremy Kemp
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test illustrates #1111: hpx::threads::get_thread_data always returns zero

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>

struct thread_data
{
    int thread_num;
};

int get_thread_num()
{
    hpx::threads::thread_id_type thread_id = hpx::threads::get_self_id();
    thread_data *data = reinterpret_cast<thread_data*>(
        hpx::threads::get_thread_data(thread_id));
    HPX_TEST(data);
    return data ? data->thread_num : 0;
}

int main()
{
    boost::scoped_ptr<thread_data> data_struct(new thread_data());
    data_struct->thread_num = 42;

    hpx::threads::thread_id_type thread_id = hpx::threads::get_self_id();
    hpx::threads::set_thread_data(thread_id,
        reinterpret_cast<std::size_t>(data_struct.get()));

    HPX_TEST_EQ(get_thread_num(), 42);

    return hpx::util::report_errors();
}

