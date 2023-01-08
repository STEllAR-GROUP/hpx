//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <utility>

template <typename T>
T generate_id()
{
    static std::atomic<T> generator(0);
    return ++generator;
}

template <typename T, std::size_t Buckets>
class static_hashed_set
{
public:
    static constexpr std::size_t calc_index(std::size_t id)
    {
        // knuth hash ... does not need to be good, but has to be portable
        std::size_t factor = std::size_t((float) Buckets * 1.616f);
        return (id * factor) % Buckets;
    }

    bool insert(T const& id)
    {
        std::size_t index = calc_index(std::size_t(id));

        std::lock_guard lock(ref_mutex[index]);

        std::pair<typename std::set<T>::iterator, bool> p =
            data[index].insert(id);

        return p.second;
    }

    bool find(T const& id)
    {
        std::size_t index = calc_index(std::size_t(id));

        std::lock_guard lock(ref_mutex[index]);

        return data[index].find(id) != data[index].end();
    }

    bool erase(T const& id)
    {
        std::size_t index = calc_index(std::size_t(id));

        std::lock_guard lock(ref_mutex[index]);

        if (data[index].find(id) != data[index].end())
        {
            data[index].erase(id);
            HPX_TEST(data[index].find(id) == data[index].end());
            return true;
        }
        else
        {
            return false;
        }
    }

    std::size_t count_nodes() const
    {
        std::size_t ret = 0;
        for (int i = 0; i != Buckets; ++i)
        {
            std::lock_guard lock(ref_mutex[i]);
            ret += data[i].size();
        }
        return ret;
    }

private:
    std::array<std::set<T>, Buckets> data;
    mutable std::array<hpx::mutex, Buckets> ref_mutex;
};

struct test_equal
{
    explicit constexpr test_equal(int i) noexcept
      : i(i)
    {
    }

    void operator()(int arg) const
    {
        HPX_TEST_EQ(arg, i);
    }

    int i;
};
