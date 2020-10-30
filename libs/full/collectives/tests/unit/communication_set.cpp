//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/pack_traversal.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/collectives/detail/communication_set_node.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace test {

    ///////////////////////////////////////////////////////////////////////////
    struct communication_set_tag
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // tag used to extract node the given node of the communicating set is
    // connected to
    struct get_connected_to
    {
    };

    // tag used to extract the number of the given node of the communication set
    struct get_site_number
    {
    };

    // tag used to verify all nodes are connected
    struct get_connected_to_zero
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    using get_connected_to_action =
        typename hpx::lcos::detail::communication_set_node::template get_action<
            communication_set_tag, std::size_t, get_connected_to>;

    using get_site_number_action =
        typename hpx::lcos::detail::communication_set_node::template get_action<
            communication_set_tag, std::size_t, get_site_number>;

    using get_connected_to_zero_action =
        typename hpx::lcos::detail::communication_set_node::template get_action<
            communication_set_tag, hpx::future<void>, get_connected_to_zero>;
}    // namespace test

namespace hpx { namespace traits {

    template <typename Communicator>
    struct communication_operation<Communicator, test::communication_set_tag>
      : std::enable_shared_from_this<
            communication_operation<Communicator, test::communication_set_tag>>
    {
        communication_operation(Communicator& comm)
          : communicator_(comm)
        {
        }

        template <typename Result>
        Result get(std::size_t, test::get_connected_to)
        {
            return communicator_.connect_to_;
        }

        template <typename Result>
        Result get(std::size_t, test::get_site_number)
        {
            return communicator_.site_;
        }

        template <typename Result>
        Result get(std::size_t, test::get_connected_to_zero)
        {
            // first forward request to parent
            if (communicator_.connect_to_ != communicator_.site_)
            {
                hpx::sync(test::get_connected_to_zero_action{},
                    communicator_.connected_node_.get(), communicator_.site_,
                    test::get_connected_to_zero{});
            }

            // now, handle request ourselves
            using mutex_type = typename Communicator::mutex_type;

            auto this_ = this->shared_from_this();
            auto on_ready = [this_ = std::move(this_)](
                                shared_future<void>&& f) {
                f.get();    // propagate any exceptions

                auto& communicator = this_->communicator_;

                std::unique_lock<mutex_type> l(communicator.mtx_);
                communicator.which_ = 0;
            };

            std::unique_lock<mutex_type> l(communicator_.mtx_);
            util::ignore_while_checking<std::unique_lock<mutex_type>> il(&l);

            hpx::future<void> f = communicator_.gate_.get_shared_future(l).then(
                hpx::launch::sync, on_ready);

            communicator_.gate_.synchronize(1, l);

            if (communicator_.gate_.set(communicator_.which_++, l))
            {
                HPX_ASSERT_DOESNT_OWN_LOCK(l);
            }
            return f;
        }

        Communicator& communicator_;
    };
}}    // namespace hpx::traits

///////////////////////////////////////////////////////////////////////////////
// Every node is connected to another one
/*
                 /                     \
                0                       8
               / \                     / \
              /   \                   /   \
             /     \                 /     \
            /       \               /       \
           /         \             /         \
          0           4           8           2
         / \         / \         / \         / \
        /   \       /   \       /   \       /   \
       0     2     4     6     8     0     2     4    <-- communicator nodes
      / \   / \   / \   / \   / \   / \   / \   / \
     0   1 2   3 4   5 6   7 8   9 0   1 2   3 4   5  <-- participants
*/
// clang-format off
std::size_t connected_nodes_2[] = {
    0,  0,  0,  4,  0,  8,  8, 12,
    0, 16, 16, 20, 16, 24, 24, 28,
    0, 32, 32, 36, 32, 40, 40, 44,
   32, 48, 48, 52, 48, 56, 56, 60,
};

std::size_t connected_nodes_4[] = {
    0,  0,  0,  0,  0, 16, 16, 16,
    0, 32, 32, 32,  0, 48, 48, 48,
};

std::size_t connected_nodes_8[] = {
    0,  0,  0,  0,  0,  0,  0,  0,
};

std::size_t connected_nodes_16[] = {
    0,  0,  0,  0,
};

std::size_t connected_nodes_32[] = {
    0,  0,
};

std::size_t connected_nodes_64[] = {
    0,
};
// clang-format on

///////////////////////////////////////////////////////////////////////////////
constexpr char const* communication_set_basename = "/test/communication_set";

void test_communication_set(std::size_t size, std::size_t arity)
{
    // create nodes of communication set
    std::string basename =
        hpx::util::format("{}_{}_{}", communication_set_basename, size, arity);

    std::vector<hpx::future<hpx::id_type>> nodes;
    nodes.reserve(size);
    for (std::size_t i = 0; i != size; ++i)
    {
        nodes.push_back(hpx::lcos::create_communication_set(
            basename.c_str(), size, i, arity));
    }

    // verify that all leaf-nodes are connected to the proper parent
    std::vector<hpx::id_type> node_ids = hpx::util::unwrap(nodes);
    for (std::size_t i = 0; i < size; i += arity)
    {
        for (std::size_t j = 1; j != arity && i + j < size; ++j)
        {
            HPX_TEST(node_ids[i] == node_ids[i + j]);
        }
    }

    // verify the number of the connected node for this endpoint
    for (std::size_t i = 0; i != size; ++i)
    {
        hpx::future<std::size_t> f = hpx::async(test::get_site_number_action{},
            node_ids[i], i, test::get_site_number{});
        HPX_TEST_EQ(f.get(), (i / arity) * arity);
    }

    // verify how created nodes are connected
    for (std::size_t i = 0; i != size; ++i)
    {
        hpx::future<std::size_t> f = hpx::async(test::get_connected_to_action{},
            node_ids[i], i, test::get_connected_to{});

        switch (arity)
        {
        case 2:
            HPX_TEST_EQ(f.get(), connected_nodes_2[i / 2]);
            break;

        case 4:
            HPX_TEST_EQ(f.get(), connected_nodes_4[i / 4]);
            break;

        case 8:
            HPX_TEST_EQ(f.get(), connected_nodes_8[i / 8]);
            break;

        case 16:
            HPX_TEST_EQ(f.get(), connected_nodes_16[i / 16]);
            break;

        case 32:
            HPX_TEST_EQ(f.get(), connected_nodes_32[i / 32]);
            break;

        case 64:
            HPX_TEST_EQ(f.get(), connected_nodes_64[i / 64]);
            break;
        }
    }

    // verify the number of the connected node for this endpoint
    std::vector<hpx::future<void>> futures;
    futures.reserve(size);
    for (std::size_t i = 0; i != size; ++i)
    {
        futures.push_back(hpx::async(test::get_connected_to_zero_action{},
            node_ids[i], i, test::get_connected_to_zero{}));
    }

    // rethrow exceptions
    hpx::util::unwrap(futures);
}

int hpx_main()
{
    for (std::size_t size = 0; size != 64; ++size)
    {
        test_communication_set(size + 1, 2);
        test_communication_set(size + 1, 4);
        test_communication_set(size + 1, 8);
        test_communication_set(size + 1, 16);
        test_communication_set(size + 1, 32);
        test_communication_set(size + 1, 64);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
