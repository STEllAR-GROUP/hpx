//  Copyright (c) 2011-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <hpx/components/iostreams/server/buffer.hpp>

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

namespace hpx { namespace iostreams { namespace detail
{
    struct order_output
    {
        typedef std::map<std::uint64_t, buffer> output_data_type;
        typedef std::pair<std::uint64_t, output_data_type> data_type;
        typedef std::map<std::uint32_t, data_type> output_data_map_type;

        template <typename F, typename Mutex>
        void output(std::uint32_t locality_id, std::uint64_t count,
            detail::buffer const& buf_in, F const& write_f, Mutex& mtx)
        {
            detail::buffer in(buf_in);
            std::unique_lock<Mutex> l(mtx);
            data_type& data = output_data_map_[locality_id]; //-V108

            if (count == data.first)
            {
                // this is the next expected output line
                {
                    // output the line as requested
                    util::unlock_guard<std::unique_lock<Mutex> > ul(l);
                    in.write(write_f, mtx);
                }
                ++data.first;

                // print all consecutive pending buffers
                output_data_type::iterator next = data.second.find(++count);
                while (next != data.second.end())
                {
                    buffer next_in = (*next).second;
                    {
                        // output the next line
                        util::unlock_guard<std::unique_lock<Mutex> > ul(l);
                        next_in.write(write_f, mtx);
                    }
                    data.second.erase(next);

                    // find next entry in map
                    ++data.first;
                    next = data.second.find(++count);
                }
            }
            else
            {
                HPX_ASSERT(count > data.first);
                data.second.insert(output_data_type::value_type(count, in));
            }
        }

    private:
        output_data_map_type output_data_map_;
    };
}}}



