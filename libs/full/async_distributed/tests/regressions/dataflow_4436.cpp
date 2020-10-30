//  Copyright (c) 2020 albestro
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/pack_traversal/traits/pack_traversal_rebind_container.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

namespace custom {

    template <typename T>
    struct vector : public std::vector<T>
    {
        vector() = default;
    };

}    // namespace custom

namespace hpx { namespace traits {

    template <typename NewType, typename OldType>
    struct pack_traversal_rebind_container<NewType, custom::vector<OldType>>
    {
        static custom::vector<NewType> call(custom::vector<OldType> const&)
        {
            // Create a new version of the container for the new data type
            return custom::vector<NewType>();
        }
    };
}}    // namespace hpx::traits

int hpx_main()
{
    custom::vector<hpx::future<int>> values_futures;

    for (int i = 1; i <= 3; ++i)
    {
        values_futures.emplace_back(hpx::make_ready_future(i));
    }

    auto f = hpx::dataflow(hpx::util::unwrapping([](const auto&& values) {
        return std::accumulate(values.begin(), values.end(), 0);
    }),
        values_futures);

    HPX_TEST_EQ(f.get(), 6);

    return hpx::finalize();
}

int main(int argc, char** argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);

    return hpx::util::report_errors();
}
#endif
