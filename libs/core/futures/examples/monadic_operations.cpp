//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/futures/monadic_operations.hpp>

#include <iostream>
#include <string>

int hpx_main()
{
    // Legacy (.then):
    auto f_legacy = hpx::make_ready_future(42)
                        .then([](auto fut) {
                            int val = fut.get();
                            return hpx::make_ready_future(std::to_string(val));
                        })
                        .unwrap();

    std::cout << "Legacy Result: " << f_legacy.get() << std::endl;

    // Modern (Monadic):
    auto f_modern = hpx::futures::and_then(hpx::make_ready_future(42), [](int val) {
        return hpx::make_ready_future(std::to_string(val));
    });

    std::cout << "Modern Result: " << f_modern.get() << std::endl;

    // Transform Example:
    auto f_transform = hpx::futures::transform(hpx::make_ready_future(42), [](int val) {
        return std::to_string(val);
    });

    std::cout << "Transform Result: " << f_transform.get() << std::endl;

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    return hpx::local::init(hpx_main, argc, argv);
}
