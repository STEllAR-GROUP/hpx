//  Copyright (c) 2014-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/debug/thread_stacktrace.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

#include <vector>
#include <sstream>
#include <string>

namespace hpx {
namespace util {
namespace debug {

    // ------------------------------------------------------------------------
    // return a vector of suspended/other task Ids
    std::vector<hpx::threads::thread_id_type> get_task_ids(
        hpx::threads::thread_state_enum state)
    {
        std::vector<hpx::threads::thread_id_type> thread_ids_vector;
        //
        hpx::threads::enumerate_threads(
            [&thread_ids_vector](hpx::threads::thread_id_type id) -> bool
            {
                thread_ids_vector.push_back(id);
                return true; // always continue enumeration
            },
            state
        );
        return thread_ids_vector;
    }

    // ------------------------------------------------------------------------
    // return a vector of thread data structure pointers for suspended tasks
    std::vector<hpx::threads::thread_data*> get_task_data(
        hpx::threads::thread_state_enum state)
    {
        std::vector<hpx::threads::thread_data*> thread_data_vector;
        //
        hpx::threads::enumerate_threads(
            [&thread_data_vector](hpx::threads::thread_id_type id) -> bool
            {
                hpx::threads::thread_data *data = id.get();
                thread_data_vector.push_back(data);
                return true; // always continue enumeration
            },
            state
        );
        return thread_data_vector;
    }

    // ------------------------------------------------------------------------
    // return string containing the stack backtrace for suspended tasks
    std::string suspended_task_backtraces()
    {
        std::vector<hpx::threads::thread_data*> tlist =
            get_task_data(hpx::threads::suspended);
        //
        std::stringstream tmp;
        //
        int count = 0;
        for (const auto & data : tlist) {
            tmp << "Stack trace "
                << "" << std::dec << count << " : "
                << "0x" << std::setfill('0') << std::setw(12) << std::noshowbase
                << std::hex << (uintptr_t)(data) << " : \n"
#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
                << data->backtrace() << "\n\n";
#else
                << "Enable HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION in CMake";
#endif
            ++count;
        }
        return tmp.str();
    }
}}}
