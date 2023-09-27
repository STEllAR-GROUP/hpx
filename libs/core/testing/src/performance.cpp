//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define ANKERL_NANOBENCH_IMPLEMENT
#include <hpx/testing/performance.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <nanobench.h>

#define NANOBENCH_EPOCHS 24
#define NANOBENCH_WARMUP 40

namespace hpx::util {

    namespace detail {

        char const* nanobench_hpx_template() noexcept
        {
            return R"DELIM({
    "outputs": [
{{#result}}        {
            "name": "{{name}}",
            "executor": "{{context(executor)}}",
            "series": [
                {{#measurement}}{{elapsed}}{{^-last}},
                {{/-last}}{{/measurement}}
            ]
        }{{^-last}},{{/-last}}
{{/result}}    ]
})DELIM";
        }

        ankerl::nanobench::Bench& bench()
        {
            static ankerl::nanobench::Bench b;
            static ankerl::nanobench::Config cfg;

            cfg.mWarmup = NANOBENCH_WARMUP;
            cfg.mNumEpochs = NANOBENCH_EPOCHS;

            return b.config(cfg);
        }

    }    // namespace detail

    void perftests_report(std::string const& name, std::string const& exec,
        std::size_t const steps, hpx::function<void()>&& test)
    {
        if (steps == 0)
            return;

        std::size_t const steps_per_epoch = steps / NANOBENCH_EPOCHS + 1;

        detail::bench()
            .name(name)
            .context("executor", exec)
            .minEpochIterations(steps_per_epoch)
            .run(test);
    }

    void perftests_print_times(std::ostream& strm, char const* templ)
    {
        if (!templ)
            templ = detail::nanobench_hpx_template();

        detail::bench().render(templ, strm);
    }
}    // namespace hpx::util
