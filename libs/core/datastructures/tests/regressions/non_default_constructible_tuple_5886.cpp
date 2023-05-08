//  Copyright (c) 2022 Julien Esseiva
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>

#include <hpx/datastructures/serialization/tuple.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

class event
{
    std::vector<int> data;
    std::string name;

    friend class hpx::serialization::access;

    template <typename Archive>
    constexpr void serialize(Archive&, const unsigned int) noexcept
    {
        // nothing else to do
    }

    template <typename Archive>
    friend void save_construct_data(
        Archive& ar, event const* t, const unsigned int)
    {
        ar & t->name & t->data;
    }

    template <typename Archive>
    friend void load_construct_data(Archive& ar, event* t, const unsigned int)
    {
        std::string name;
        std::vector<int> data;

        // clang-format off
        ar & name & data;
        // clang-format on

        ::new (t) event(std::move(name), std::move(data));
    }

public:
    event() = delete;

    event(std::string const& name, std::size_t n_elem)
      : data(n_elem)
      , name(name)
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> distr(0, 10);
        for (auto it = data.begin(); it != data.end(); ++it)
        {
            *it = distr(gen);
        }
    }

    event(std::string&& name, std::vector<int>&& data) noexcept
      : data(std::move(data))
      , name(std::move(name))
    {
    }

    friend bool operator==(event const& lhs, event const& rhs) noexcept
    {
        return lhs.name == rhs.name && lhs.data == rhs.data;
    }
};

int hpx_main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    auto datao = std::make_shared<hpx::tuple<event, event>>(
        event("evtid", 15), event("evtid", 15));

    oarchive << datao;

    hpx::serialization::input_archive iarchive(buffer);
    std::shared_ptr<hpx::tuple<event, event>> datai;

    iarchive >> datai;

    HPX_TEST(*datai == *datao);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
