//  Copyright (c) 2005-2012 Hartmut Kaiser
//  Copyright (c) 2014      Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/hardware/timestamp.hpp>

#include <cstdint>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    //
    //  tick_counter - a timer
    //
    ///////////////////////////////////////////////////////////////////////////
    class tick_counter
    {
    public:
        tick_counter(std::uint64_t& output)
          : start_time_(take_time_stamp())
          , output_(output)
        {
        }

        ~tick_counter()
        {
            output_ += take_time_stamp() - start_time_;
        }

    protected:
        static std::uint64_t take_time_stamp()
        {
            return hpx::util::hardware::timestamp();
        }

    private:
        std::uint64_t const start_time_;
        std::uint64_t& output_;
    };
}}    // namespace hpx::util
