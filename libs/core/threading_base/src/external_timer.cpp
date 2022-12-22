//  Copyright (c) 2007-2013 Kevin Huck
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <hpx/config.hpp>

#ifdef HPX_HAVE_APEX
#include <hpx/assert.hpp>
#include <hpx/threading_base/external_timer.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace hpx::util {

    static enable_parent_task_handler_type enable_parent_task_handler;

    void set_enable_parent_task_handler(enable_parent_task_handler_type f)
    {
        enable_parent_task_handler = HPX_MOVE(f);
    }

    namespace external_timer {

        std::shared_ptr<task_wrapper> new_task(
            threads::thread_description const& description,
            std::uint32_t /* parent_locality_id */,
            threads::thread_id_type parent_task)
        {
            std::shared_ptr<task_wrapper> parent_wrapper;

            // Parent pointers aren't reliable in distributed runs.
            if (parent_task != nullptr && enable_parent_task_handler &&
                enable_parent_task_handler())
            {
                parent_wrapper =
                    get_thread_id_data(parent_task)->get_timer_data();
            }

            if (description.kind() ==
                threads::thread_description::data_type_description)
            {
                return new_task(
                    description.get_description(), UINTMAX_MAX, parent_wrapper);
            }
            else
            {
                HPX_ASSERT(description.kind() ==
                    threads::thread_description::data_type_address);
                return new_task(
                    description.get_address(), UINTMAX_MAX, parent_wrapper);
            }
        }

        // register the function pointers
        void register_external_timer(registration& reg)
        {
            switch (reg.type)
            {
            case init_flag:
                init_function = reg.record.init;
                break;

            case finalize_flag:
                finalize_function = reg.record.finalize;
                break;

            case register_thread_flag:
                register_thread_function = reg.record.register_thread;
                break;

            case new_task_string_flag:
                new_task_string_function = reg.record.new_task_string;
                break;

            case new_task_address_flag:
                new_task_address_function = reg.record.new_task_address;
                break;

            case update_task_string_flag:
                update_task_string_function = reg.record.update_task_string;
                break;

            case update_task_address_flag:
                update_task_address_function = reg.record.update_task_address;
                break;

            case sample_value_flag:
                sample_value_function = reg.record.sample_value;
                break;

            case send_flag:
                send_function = reg.record.send;
                break;

            case recv_flag:
                recv_function = reg.record.recv;
                break;

            case start_flag:
                start_function = reg.record.start;
                break;

            case stop_flag:
                stop_function = reg.record.stop;
                break;

            case yield_flag:
                yield_function = reg.record.yield;
                break;
            }
        }

        // Instantiate the function pointers.
        init_t* init_function{nullptr};
        finalize_t* finalize_function{nullptr};
        register_thread_t* register_thread_function{nullptr};
        new_task_string_t* new_task_string_function{nullptr};
        new_task_address_t* new_task_address_function{nullptr};
        sample_value_t* sample_value_function{nullptr};
        send_t* send_function{nullptr};
        recv_t* recv_function{nullptr};
        update_task_string_t* update_task_string_function{nullptr};
        update_task_address_t* update_task_address_function{nullptr};
        start_t* start_function{nullptr};
        stop_t* stop_function{nullptr};
        yield_t* yield_function{nullptr};
    }    // namespace external_timer
}    // namespace hpx::util

#endif
