//  Copyright (c) 2021-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/serialization.hpp>

#if defined(HPX_SERIALIZATION_HAVE_ALLOW_RAW_POINTER_SERIALIZATION)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/tuple.hpp>

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
#if defined(HPX_SERIALIZATION_HAVE_ALLOW_CONST_TUPLE_MEMBERS)
    using tuple_type = hpx::tuple<int*, int const>;
#else
    using tuple_type = hpx::tuple<int*, int>;
#endif

    // serialize raw pointer as part of tuple
    {
        // enforce pointers being serializable
        static_assert(hpx::traits::is_bitwise_serializable_v<tuple_type>);

        int value = 42;
        tuple_type const ot{&value, value};

        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        oarchive << ot;
        std::size_t const size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        tuple_type it{nullptr, 0};
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

#else

int main(int, char*[])
{
    return 0;
}

#endif
