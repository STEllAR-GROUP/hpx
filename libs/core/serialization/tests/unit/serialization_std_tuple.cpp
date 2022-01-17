//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/std_tuple.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

template <typename T>
struct A
{
    A() = default;

    explicit A(T t)
      : t_(t)
    {
    }
    T t_;

    A& operator=(T t)
    {
        t_ = t;
        return *this;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        // clang-format off
        ar & t_;
        // clang-format on
    }

    friend bool operator==(A const& lhs, A const& rhs)
    {
        return lhs.t_ == rhs.t_;
    }
};

void test()
{
    {
        std::tuple<int, double, std::string, A<int>> ot{
            42, 42.0, "42.0", A<int>{0}};

        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::tuple<int, double, std::string, A<int>> it;
        iarchive >> it;
        HPX_TEST(ot == it);
    }
    {
        std::tuple<> ot{};

        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::tuple<> it;
        iarchive >> it;
        HPX_TEST(ot == it);
    }
}

int main()
{
    test();
    return hpx::util::report_errors();
}
