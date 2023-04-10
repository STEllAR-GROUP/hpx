//  Copyright (c) 2021-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// enforce pointers being serializable
#define HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION

// allow for const tuple members
#define HPX_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS

#include <hpx/config.hpp>
#include <hpx/init.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/tuple.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <string>
#include <vector>

// non-bitwise copyable type
struct A
{
    A() = default;

    template <typename Archive>
    void serialize(Archive&, unsigned)
    {
    }

    friend bool operator==(A const&, A const&)
    {
        return true;
    }
};

int hpx_main()
{
    // serialize raw pointer as part of tuple
    {
        static_assert(hpx::traits::is_bitwise_serializable_v<
            hpx::tuple<int*, int const>>);

        int value = 42;
        hpx::tuple<int*, int const> ot{&value, value};

        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        hpx::tuple<int*, int const> it{nullptr, 0};
        iarchive >> it;
        HPX_TEST(ot == it);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
