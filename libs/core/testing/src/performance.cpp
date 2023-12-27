//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/testing/performance.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(HPX_FETCH_NANOBENCH) || defined(HPX_NANOBENCH_ROOT)
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#endif

namespace hpx::util {

    namespace detail {

#if defined(HPX_FETCH_NANOBENCH) || defined(HPX_NANOBENCH_ROOT)
        constexpr int nanobench_epochs = 24;
        constexpr int nanobench_warmup = 40;

        char const* nanobench_hpx_template() constexpr noexcept
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

            cfg.mWarmup = nanobench_warmup;
            cfg.mNumEpochs = nanobench_epochs;

            return b.config(cfg);
        }
#else
        // Json output for performance reports
        class json_perf_times
        {
            using key_t = std::tuple<std::string, std::string>;
            using value_t = std::vector<double>;
            using map_t = std::map<key_t, value_t>;

            map_t m_map;

            HPX_CORE_EXPORT friend std::ostream& operator<<(
                std::ostream& strm, json_perf_times const& obj);

        public:
            HPX_CORE_EXPORT void add(std::string const& name,
                std::string const& executor, double time);
        };

        json_perf_times& times()
        {
            static json_perf_times res;
            return res;
        }

        void add_time(std::string const& test_name, std::string const& executor,
            double time)
        {
            times().add(test_name, executor, time);
        }

        std::ostream& operator<<(std::ostream& strm, json_perf_times const& obj)
        {
            strm << "{\n";
            strm << "  \"outputs\" : [";
            int outputs = 0;
            for (auto&& item : obj.m_map)
            {
                if (outputs)
                    strm << ",";
                strm << "\n    {\n";
                strm << R"(      "name" : ")" << std::get<0>(item.first)
                     << "\",\n";
                strm << R"(      "executor" : ")" << std::get<1>(item.first)
                     << "\",\n";
                strm << R"(      "series" : [)";
                double average = 0.0;
                int series = 0;
                for (auto const val : item.second)
                {
                    if (series)
                        strm << ", ";
                    strm << val;
                    ++series;
                    average += val;
                }
                strm << "],\n";
                strm << "      \"average\" : " << average / series << "\n";
                strm << "    }";
                ++outputs;
            }
            if (outputs)
                strm << "\n  ";
            strm << "]\n";
            strm << "}\n";
            return strm;
        }

        void json_perf_times::add(
            std::string const& name, std::string const& executor, double time)
        {
            m_map[key_t(name, executor)].push_back(time);
        }
#endif

    }    // namespace detail

#if defined(HPX_FETCH_NANOBENCH) || defined(HPX_NANOBENCH_ROOT)
    void perftests_report(std::string const& name, std::string const& exec,
        std::size_t const steps, hpx::function<void()>&& test)
    {
        if (steps == 0)
            return;

        std::size_t const steps_per_epoch = steps / nanobench_epochs + 1;

        detail::bench()
            .name(name)
            .context("executor", exec)
            .minEpochIterations(steps_per_epoch)
            .run(test);
    }

    // Print all collected results to the provided stream,
    // formatted the json according to the provided
    // "mustache-style" template
    void perftests_print_times(std::ostream& strm, char const* templ)
    {
        detail::bench().render(templ, strm);
    }

    // Overload that uses a default nanobench template
    void perftests_print_times(std::ostream& strm)
    {
        perftests_print_times(detail::nanobench_hpx_template(), strm);
    }

    // Overload that uses a default nanobench template and prints to std::cout
    void perftests_print_times()
    {
        perftests_print_times(detail::nanobench_hpx_template(), std::cout);
    }
#else
    void perftests_report(std::string const& name, std::string const& exec,
        std::size_t const steps, hpx::function<void()>&& test)
    {
        if (steps == 0)
            return;

        // First iteration to cache the data
        test();
        using timer = std::chrono::high_resolution_clock;
        for (size_t i = 0; i != steps; ++i)
        {
            // For now we don't flush the cache
            //flush_cache();
            timer::time_point start = timer::now();
            test();
            // default is in seconds
            auto time =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    timer::now() - start);
            detail::add_time(name, exec, time.count());
        }
    }

    void perftests_print_times()
    {
        std::cout << detail::times();
    }
#endif
}    // namespace hpx::util
