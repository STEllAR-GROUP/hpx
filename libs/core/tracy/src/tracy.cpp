//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_MODULE_TRACY)
#include <hpx/tracy/tracy.hpp>

#include <string>

#include <tracy/Tracy.hpp>

namespace hpx::tracy {

    void set_thread_name(char const* name) noexcept
    {
        ::tracy::SetThreadName(name);
    }

    // Expose Tracy fibers support
    void enter_fiber(char const* name) noexcept
    {
        ::TracyFiberEnter(name);
    }

    void leave_fiber() noexcept
    {
        ::TracyFiberLeave;
    }

    // Create a new plot in Tracy
    void create_counter(std::string const& name) noexcept
    {
        ::TracyPlotConfig(
            name.c_str(), ::tracy::PlotFormatType::Number, true, false, 0);
    }

    // Pass a plot value to Tracy
    void sample_value(std::string const& name, double const value) noexcept
    {
        ::TracyPlot(name.c_str(), value);
    }
}    // namespace hpx::tracy

#endif
