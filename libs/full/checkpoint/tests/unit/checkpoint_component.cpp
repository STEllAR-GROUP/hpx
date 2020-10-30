// Copyright (c) 2018 Adrian Serio
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example tests the interoperability of save_checkpoint and
// restore_checkpoint with components.

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>

#include <hpx/include/components.hpp>
#include <hpx/modules/checkpoint.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/serialization/shared_ptr.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using hpx::util::checkpoint;
using hpx::util::prepare_checkpoint;
using hpx::util::restore_checkpoint;
using hpx::util::save_checkpoint;

struct data_server : hpx::components::component_base<data_server>
{
    data_server() = default;
    ~data_server() = default;

    data_server(std::vector<int>&& data)
      : data_(std::move(data))
    {
    }

    std::vector<int> get_data()
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(data_server, get_data, get_data_action);

    void print()
    {
        for (size_t i = 0; i < data_.size(); i++)
        {
            std::cout << data_[i];
            if (data_.size() - 1 != i)
            {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    }
    HPX_DEFINE_COMPONENT_ACTION(data_server, print, print_action);

    std::vector<int> data_;

    // Serialization Definition
    friend class hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& arch, const unsigned int /* version */)
    {
        // clang-format off
        arch & data_;
        // clang-format on
    }
};

using data_server_type = hpx::components::component<data_server>;
HPX_REGISTER_COMPONENT(data_server_type, data_server);

HPX_REGISTER_ACTION(data_server::get_data_action);
HPX_REGISTER_ACTION(data_server::print_action);

struct data_client : hpx::components::client_base<data_client, data_server>
{
    using base_type = hpx::components::client_base<data_client, data_server>;
    data_client() = default;
    ~data_client() = default;

    data_client(hpx::id_type where, std::vector<int>&& data)
      : base_type(hpx::new_<data_server>(hpx::colocated(where), data))
    {
    }

    data_client(hpx::id_type&& id)
      : base_type(std::move(id))
    {
    }

    data_client(hpx::shared_future<hpx::id_type> const& id)
      : base_type(id)
    {
    }

    hpx::future<std::vector<int>> get_data()
    {
        data_server::get_data_action get_act;
        return hpx::async(get_act, get_id());
    }

    hpx::future<void> print()
    {
        data_server::print_action print_act;
        return hpx::async(print_act, get_id());
    }
};

int main()
{
    // Test 1
    //[shared_ptr_example
    // test checkpoint a component using a shared_ptr
    std::vector<int> vec{1, 2, 3, 4, 5};
    data_client A(hpx::find_here(), std::move(vec));

    // Checkpoint Server
    hpx::id_type old_id = A.get_id();

    hpx::future<std::shared_ptr<data_server>> f_a_ptr =
        hpx::get_ptr<data_server>(A.get_id());
    std::shared_ptr<data_server> a_ptr = f_a_ptr.get();
    hpx::future<checkpoint> f = save_checkpoint(a_ptr);
    auto&& data = f.get();

    // test prepare_checkpoint API
    checkpoint c = prepare_checkpoint(hpx::launch::sync, a_ptr);
    HPX_TEST(c.size() == data.size());

    // Restore Server
    // Create a new server instance
    std::shared_ptr<data_server> b_server;
    restore_checkpoint(data, b_server);
    //]

    HPX_TEST(A.get_data().get() == b_server->get_data());

    // Re-create Client
    data_client B(std::move(old_id));

    HPX_TEST(A.get_data().get() == B.get_data().get());

    // Test 2
    //[client_example
    // Try to checkpoint and restore a component with a client
    std::vector<int> vec3{10, 10, 10, 10, 10};

    // Create a component instance through client constructor
    data_client D(hpx::find_here(), std::move(vec3));
    hpx::future<checkpoint> f3 = save_checkpoint(D);

    // Create a new client
    data_client E;

    // Restore server inside client instance
    restore_checkpoint(f3.get(), E);
    //]

    HPX_TEST(D.get_data().get() == E.get_data().get());

    return hpx::util::report_errors();
}
#endif
