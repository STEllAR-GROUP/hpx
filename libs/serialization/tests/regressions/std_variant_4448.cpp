//  Copyright (c) 2020 Yorlik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/serialization.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

struct thing
{
    std::variant<int, double, std::string> v;
    std::variant<std::string, double, std::int64_t> w;

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        // clang-format off
        ar & v;
        ar & w;
        // clang-format on
    }
};
#endif

int main(int argc, char* argv[])
{
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
    std::vector<std::byte> cont;
    hpx::serialization::output_archive ar(cont);

    thing X;
    ar << X;
#endif

    return 0;
}
