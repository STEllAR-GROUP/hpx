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

#if defined(HPX_HAVE_NANOBENCH)
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#endif

namespace hpx::util {

    void perftests_cfg(hpx::program_options::options_description& cmdline)
    {
        cmdline.add_options()("hpx:detailed_bench",
            "Use if detailed benchmarks are required, showing the execution "
            "time taken for each epoch")("hpx:print_cdash_img_path",
            "Print the path to the images to be uploaded, in CDash XML format");
    }

    void perftests_init(const hpx::program_options::variables_map& vm,
        const std::string test_name)
    {
        if (vm.count("hpx:detailed_bench"))
        {
            detailed_ = true;
        }
        if (vm.count("hpx:print_cdash_img_path"))
        {
            print_cdash_img = true;
        }
        test_name_ = test_name;
    }

    namespace detail {

#if defined(HPX_HAVE_NANOBENCH)
        constexpr int nanobench_warmup = 40;

        char const* nanobench_hpx_simple_template() noexcept
        {
            return R"DELIM(Results:
{{#result}}        
name: {{name}},
executor: {{context(executor)}},
average: {{average(elapsed)}}
{{/result}})DELIM";
        }

        char const* nanobench_hpx_cdash_template() noexcept
        {
            return R"DELIM(Results:
{{#result}}        
name: {{name}},
executor: {{context(executor)}},
average: {{average(elapsed)}}
<CTestMeasurement type="numeric/double" name="{{name}}_{{context(executor)}}">{{average(elapsed)}}</CTestMeasurement>
{{/result}})DELIM";
        }

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
}
)DELIM";
        }

        ankerl::nanobench::Bench& bench()
        {
            static ankerl::nanobench::Bench b;
            static ankerl::nanobench::Config cfg;

            cfg.mWarmup = nanobench_warmup;

            return b.config(cfg);
        }
#else
        // Json output for performance reports
        class json_perf_times
        {
            using key_t = std::tuple<std::string, std::string>;
            using value_t = std::vector<long double>;
            using map_t = std::map<key_t, value_t>;

            map_t m_map;

            HPX_CORE_EXPORT friend std::ostream& operator<<(
                std::ostream& strm, json_perf_times const& obj);

        public:
            HPX_CORE_EXPORT void add(std::string const& name,
                std::string const& executor, long double time);
        };

        json_perf_times& times()
        {
            static json_perf_times res;
            return res;
        }

        void add_time(std::string const& test_name, std::string const& executor,
            long double time)
        {
            times().add(test_name, executor, time);
        }

        std::ostream& operator<<(std::ostream& strm, json_perf_times const& obj)
        {
            if (detailed_)
            {
                strm << "{\n";
                strm << "  \"outputs\" : [";
                int outputs = 0;
                for (auto&& item : obj.m_map)
                {
                    long double average = static_cast<long double>(0.0);
                    if (outputs)
                        strm << ",";
                    strm << "\n    {\n";
                    strm << R"(      "name": ")" << std::get<0>(item.first)
                         << "\",\n";
                    strm << R"(      "executor": ")" << std::get<1>(item.first)
                         << "\",\n";
                    strm << R"(      "series": [)"
                         << "\n";
                    int series = 0;
                    strm.precision(
                        std::numeric_limits<long double>::max_digits10 - 1);
                    for (long double const val : item.second)
                    {
                        if (series)
                        {
                            strm << ",\n";
                        }
                        strm << R"(         )" << std::scientific << val;
                        ++series;
                        average += val;
                    }
                    strm << "\n       ],\n";
                    strm << std::scientific << R"(      "average": )"
                         << average / series << "\n";
                    strm << "    }";
                    ++outputs;
                }
                if (outputs)
                    strm << "\n";
                strm << "]\n";
                strm << "}\n";
            }
            else if (print_cdash_img)
            {
                strm << "Results:\n\n";
                for (auto&& item : obj.m_map)
                {
                    long double average = static_cast<long double>(0.0);
                    int series = 0;
                    strm << "name: " << std::get<0>(item.first) << "\n";
                    strm << "executor: " << std::get<1>(item.first) << "\n";
                    for (long double const val : item.second)
                    {
                        ++series;
                        average += val;
                    }
                    strm.precision(
                        std::numeric_limits<long double>::max_digits10 - 1);
                    strm << std::scientific << "average: " << average / series
                         << "\n";
                    strm << "<CTestMeasurement type=\"numeric/double\" name=\""
                         << std::get<0>(item.first) << "_"
                         << std::get<1>(item.first) << "\">" << std::scientific
                         << average / series << "</CTestMeasurement>\n\n";
                }
                for (std::size_t i = 0; i < obj.m_map.size(); i++)
                    strm << "<CTestMeasurementFile type=\"image/png\" "
                            "name=\"perftest\" >"
                         << "./" << test_name_ << "_" << i
                         << ".png</CTestMeasurementFile>\n";
            }
            else
            {
                strm << "Results:\n\n";
                for (auto&& item : obj.m_map)
                {
                    long double average = static_cast<long double>(0.0);
                    int series = 0;
                    strm << "name: " << std::get<0>(item.first) << "\n";
                    strm << "executor: " << std::get<1>(item.first) << "\n";
                    for (long double const val : item.second)
                    {
                        ++series;
                        average += val;
                    }
                    strm.precision(
                        std::numeric_limits<long double>::max_digits10 - 1);
                    strm << std::scientific << "average: " << average / series
                         << "\n\n";
                }
            }
            return strm;
        }

        void json_perf_times::add(std::string const& name,
            std::string const& executor, long double time)
        {
            m_map[key_t(name, executor)].push_back(time);
        }
#endif

    }    // namespace detail

#if defined(HPX_HAVE_NANOBENCH)
    void perftests_report(std::string const& name, std::string const& exec,
        std::size_t const steps, hpx::function<void()>&& test)
    {
        if (steps == 0)
            return;

        detail::bench()
            .name(name)
            .context("executor", exec)
            .epochs(steps)
            .run(test);
    }

    // Print all collected results to the provided stream,
    // formatted the json according to the provided
    // "mustache-style" template
    void perftests_print_times(char const* templ, std::ostream& strm)
    {
        detail::bench().render(templ, strm);
        if (!detailed_ && print_cdash_img)
        {
            for (long unsigned int i = 0; i < detail::bench().results().size();
                 i++)
            {
                strm << "<CTestMeasurementFile type=\"image/png\" "
                        "name=\"perftest\">"
                     << "./" << test_name_ << "_" << i
                     << ".png</CTestMeasurementFile>\n";
            }
        }
    }

    // Overload that uses a default nanobench template
    void perftests_print_times(std::ostream& strm)
    {
        perftests_print_times(detail::nanobench_hpx_template(), strm);
    }

    // Overload that uses a default nanobench template and prints to std::cout
    void perftests_print_times()
    {
        if (detailed_)
            perftests_print_times(detail::nanobench_hpx_template(), std::cout);
        else if (print_cdash_img)
            perftests_print_times(
                detail::nanobench_hpx_cdash_template(), std::cout);
        else
            perftests_print_times(
                detail::nanobench_hpx_simple_template(), std::cout);
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
        for (std::size_t i = 0; i != steps; ++i)
        {
            // For now we don't flush the cache
            //flush_cache();
            timer::time_point start = timer::now();
            test();
            // default is in seconds
            auto time =
                std::chrono::duration_cast<std::chrono::duration<long double>>(
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
