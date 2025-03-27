//  Copyright (c) 2024 Jiakun Yan
//  Copyright (c) 2024 Marco Diers
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/include/util.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
const std::size_t nbytes_default = (std::numeric_limits<int>::max)();
const std::size_t nbytes_add_default = 0;

struct config_t
{
    size_t nbytes;
    size_t nbytes_add;
} config;
///////////////////////////////////////////////////////////////////////////////
class Data
{
public:
    Data() = default;
    Data(std::size_t size)
      : _data(size, 'a')
    {
    }
    auto size() const
    {
        return _data.size();
    }

    char& operator[](size_t idx)
    {
        return _data[idx];
    }

    char operator[](size_t idx) const
    {
        return _data[idx];
    }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        // clang-format off
        ar & _data;
        // clang-format on
    }

private:
    std::vector<char> _data{};
};

class Component : public hpx::components::component_base<Component>
{
public:
    Component() = default;

    auto call(Data data) -> void
    {
        std::cout << "Data size: " << data.size() << '\n';
        bool flag = true;
        size_t idx = 0;
        for (; idx < data.size(); ++idx)
        {
            if (data[idx] != 'a')
            {
                flag = false;
                break;
            }
        }
        if (!flag)
            std::cout << "Data[" << idx << "] = " << data[idx]
                      << " instead of a\n";
        else
            std::cout << "data is correct\n";
        HPX_TEST_EQ(flag, true);
        return;
    }

    HPX_DEFINE_COMPONENT_ACTION(Component, call)
};

HPX_REGISTER_COMPONENT(hpx::components::component<Component>, Component);
HPX_REGISTER_ACTION(Component::call_action);

class ComponentClient
  : public hpx::components::client_base<ComponentClient, Component>
{
    using BaseType = hpx::components::client_base<ComponentClient, Component>;

public:
    template <typename... Arguments>
    ComponentClient(Arguments... arguments)
      : BaseType(std::move(arguments)...)
    {
    }

    template <typename... Arguments>
    auto call(Arguments... arguments)
    {
        return hpx::async<Component::call_action>(
            this->get_id(), std::move(arguments)...);
    }
};

int hpx_main(hpx::program_options::variables_map& b_arg)
{
    config.nbytes = b_arg["nbytes"].as<std::size_t>();
    config.nbytes_add = b_arg["nbytes-add"].as<std::size_t>();

    std::vector<ComponentClient> clients;
    auto localities(hpx::find_remote_localities());
    std::transform(std::begin(localities), std::end(localities),
        std::back_inserter(clients),
        [](auto& loc) { return hpx::new_<ComponentClient>(loc); });

    Data data(config.nbytes + config.nbytes_add);
    std::vector<decltype(clients.front().call(data))> calls;
    for (auto& client : clients)
    {
        calls.emplace_back(client.call(data));
    }
    hpx::wait_all(calls);

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    namespace po = hpx::program_options;
    po::options_description description("HPX big parcel test");

    description.add_options()("nbytes",
        po::value<std::size_t>()->default_value(nbytes_default),
        "number of bytes to send")("nbytes-add",
        po::value<std::size_t>()->default_value(nbytes_add_default),
        "number of additional bytes to send");

    hpx::init_params init_args;
    init_args.desc_cmdline = description;

    // Initialize and run HPX
    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}
#endif
