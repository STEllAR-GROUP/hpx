//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

int hpx_main()
{
    // -----------------------------------------------------------
    // and_then: automatic flattening (no .unwrap() needed)
    //
    //   Legacy:   fut.then(F).unwrap()   -- F returns future<T>
    //   Monadic:  hpx::futures::and_then(fut, F)  -- automatic unwrap
    // -----------------------------------------------------------
    auto f_legacy = hpx::make_ready_future(42)
                        .then([](auto fut) {
                            int val = fut.get();
                            return hpx::make_ready_future(std::to_string(val));
                        })
                        .unwrap();

    auto f_monadic = hpx::futures::and_then(hpx::make_ready_future(42),
        [](int val) { return hpx::make_ready_future(std::to_string(val)); });

    std::cout << "Legacy  and_then: " << f_legacy.get() << std::endl;
    std::cout << "Monadic and_then: " << f_monadic.get() << std::endl;

    // -----------------------------------------------------------
    // or_else: declarative error recovery (no try/catch)
    //
    //   Legacy:   try { fut.get(); } catch(...) { ... }
    //   Monadic:  hpx::futures::or_else(fut, recovery_fn)
    // -----------------------------------------------------------
    auto f_err = hpx::make_exceptional_future<int>(
        std::runtime_error("something failed"));

    auto f_recovered = hpx::futures::or_else(std::move(f_err),
        [](std::exception_ptr const&) { return hpx::make_ready_future(-1); });

    std::cout << "or_else recovery: " << f_recovered.get() << std::endl;

    // -----------------------------------------------------------
    // transform: apply a pure function to a future's value
    // -----------------------------------------------------------
    auto f_transformed = hpx::futures::transform(hpx::make_ready_future(42),
        [](int val) { return std::to_string(val); });

    std::cout << "transform: " << f_transformed.get() << std::endl;

    return hpx::local::finalize();
}

int main(int argc, char** argv)
{
    return hpx::local::init(hpx_main, argc, argv);
}
