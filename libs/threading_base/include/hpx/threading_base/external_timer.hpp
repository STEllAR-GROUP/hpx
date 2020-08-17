//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once    // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/threading_base/thread_description.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace hpx { namespace util {

#ifdef HPX_HAVE_APEX

    using enable_parent_task_handler_type = util::function_nonser<bool()>;

    HPX_EXPORT void set_enable_parent_task_handler(
        enable_parent_task_handler_type f);

    namespace external_timer {

        // HPX provides a smart pointer to a data object that maintains
        // information about an hpx_thread. Any library (i.e. APEX) that wants
        // to use this callback API needs to extend this class.
        struct task_wrapper
        {
        };

        // Enumeration of function type flags
        typedef enum functions_t
        {
            init_flag = 0,
            finalize_flag,
            register_thread_flag,
            new_task_string_flag,
            new_task_address_flag,
            update_task_string_flag,
            update_task_address_flag,
            sample_value_flag,
            send_flag,
            recv_flag,
            start_flag,
            stop_flag,
            yield_flag
        } functions_t;

        // Typedefs of function pointers
        typedef uint64_t init_t(const char*, const uint64_t, const uint64_t);
        typedef void finalize_t(void);
        typedef void register_thread_t(const std::string&);
        typedef std::shared_ptr<task_wrapper> new_task_string_t(
            const std::string&, const uint64_t,
            const std::shared_ptr<task_wrapper>);
        typedef std::shared_ptr<task_wrapper> new_task_address_t(
            uintptr_t, const uint64_t, const std::shared_ptr<task_wrapper>);
        typedef void sample_value_t(const std::string&, double);
        typedef void send_t(uint64_t, uint64_t, uint64_t);
        typedef void recv_t(uint64_t, uint64_t, uint64_t, uint64_t);
        typedef std::shared_ptr<task_wrapper> update_task_string_t(
            std::shared_ptr<task_wrapper>, const std::string&);
        typedef std::shared_ptr<task_wrapper> update_task_address_t(
            std::shared_ptr<task_wrapper>, uintptr_t);
        typedef void start_t(std::shared_ptr<task_wrapper>);
        typedef void stop_t(std::shared_ptr<task_wrapper>);
        typedef void yield_t(std::shared_ptr<task_wrapper>);

        // Structure for compiler type-checking of function pointer assignment
        typedef struct registration
        {
            functions_t type;
            union
            {
                init_t* init;
                finalize_t* finalize;
                register_thread_t* register_thread;
                new_task_string_t* new_task_string;
                new_task_address_t* new_task_address;
                update_task_string_t* update_task_string;
                update_task_address_t* update_task_address;
                sample_value_t* sample_value;
                send_t* send;
                recv_t* recv;
                start_t* start;
                stop_t* stop;
                yield_t* yield;
            } record;
        } registration_t;

        // The actual function pointers. Some of them need to be exported,
        // because through the miracle of chained headers they get referenced
        // outside of the HPX library.
        extern init_t* init_function;
        extern finalize_t* finalize_function;
        extern register_thread_t* register_thread_function;
        extern new_task_string_t* new_task_string_function;
        extern new_task_address_t* new_task_address_function;
        extern sample_value_t* sample_value_function;
        extern send_t* send_function;
        extern recv_t* recv_function;
        HPX_EXPORT extern update_task_string_t* update_task_string_function;
        HPX_EXPORT extern update_task_address_t* update_task_address_function;
        HPX_EXPORT extern start_t* start_function;
        HPX_EXPORT extern stop_t* stop_function;
        HPX_EXPORT extern yield_t* yield_function;

        // The function registration interface
        HPX_EXPORT void register_external_timer(
            registration_t& registration_record);

        // The actual API. For all cases, check if the function pointer is null,
        // and if not null call the registered function.
        static inline uint64_t init(const char* thread_name,
            const uint64_t comm_rank, const uint64_t comm_size)
        {
            return (init_function == nullptr) ?
                0ULL :
                init_function(thread_name, comm_rank, comm_size);
        }
        static inline void finalize(void)
        {
            if (finalize_function != nullptr)
            {
                finalize_function();
            }
        }
        static inline void register_thread(const std::string& name)
        {
            if (register_thread_function != nullptr)
            {
                register_thread_function(name);
            }
        }
        static inline std::shared_ptr<task_wrapper> new_task(
            const std::string& name, const uint64_t task_id,
            const std::shared_ptr<task_wrapper> parent_task)
        {
            return (new_task_string_function == nullptr) ?
                0ULL :
                new_task_string_function(name, task_id, parent_task);
        }
        static inline std::shared_ptr<task_wrapper> new_task(uintptr_t address,
            const uint64_t task_id,
            const std::shared_ptr<task_wrapper> parent_task)
        {
            return (new_task_address_function == nullptr) ?
                0ULL :
                new_task_address_function(address, task_id, parent_task);
        }
        static inline void send(uint64_t tag, uint64_t size, uint64_t target)
        {
            if (send_function != nullptr)
            {
                send_function(tag, size, target);
            }
        }
        static inline void recv(uint64_t tag, uint64_t size,
            uint64_t source_rank, uint64_t source_thread)
        {
            if (recv_function != nullptr)
            {
                recv_function(tag, size, source_rank, source_thread);
            }
        }
        static inline std::shared_ptr<task_wrapper> update_task(
            std::shared_ptr<task_wrapper> wrapper, const std::string& name)
        {
            return (update_task_string_function == nullptr) ?
                0ULL :
                update_task_string_function(wrapper, name);
        }
        static inline std::shared_ptr<task_wrapper> update_task(
            std::shared_ptr<task_wrapper> wrapper, uintptr_t address)
        {
            return (update_task_address_function == nullptr) ?
                0ULL :
                update_task_address_function(wrapper, address);
        }
        static inline void start(std::shared_ptr<task_wrapper> task_wrapper_ptr)
        {
            if (start_function != nullptr)
            {
                start_function(task_wrapper_ptr);
            }
        }
        static inline void stop(std::shared_ptr<task_wrapper> task_wrapper_ptr)
        {
            if (stop_function != nullptr)
            {
                stop_function(task_wrapper_ptr);
            }
        }
        static inline void yield(std::shared_ptr<task_wrapper> task_wrapper_ptr)
        {
            if (yield_function != nullptr)
            {
                yield_function(task_wrapper_ptr);
            }
        }

        HPX_EXPORT std::shared_ptr<task_wrapper> new_task(
            thread_description const& description,
            std::uint32_t parent_locality_id,
            threads::thread_id_type const& parent_task);

        HPX_EXPORT inline std::shared_ptr<task_wrapper> update_task(
            std::shared_ptr<task_wrapper> wrapper,
            thread_description const& description)
        {
            if (wrapper == nullptr)
            {
                threads::thread_id_type parent_task(nullptr);
                // doesn't matter which locality we use, the parent is null
                return new_task(description, 0, parent_task);
            }
            else if (description.kind() ==
                thread_description::data_type_description)
            {
                // Disambiguate the call by making a temporary string object
                return update_task(
                    wrapper, std::string(description.get_description()));
            }
            else
            {
                HPX_ASSERT(description.kind() ==
                    thread_description::data_type_address);
                return update_task(wrapper, description.get_address());
            }
        }

        // This is a scoped object around task scheduling to measure the time
        // spent executing hpx threads
        struct scoped_timer
        {
            explicit scoped_timer(std::shared_ptr<task_wrapper> data_ptr)
              : stopped(false)
              , data_(nullptr)
            {
                // APEX internal actions are not timed. Otherwise, we would end
                // up with recursive timers. So it's possible to have a null
                // task wrapper pointer here.
                if (data_ptr != nullptr)
                {
                    data_ = data_ptr;
                    hpx::util::external_timer::start(data_);
                }
            }
            ~scoped_timer()
            {
                stop();
            }

            void stop()
            {
                if (!stopped)
                {
                    stopped = true;
                    // APEX internal actions are not timed. Otherwise, we would
                    // end up with recursive timers. So it's possible to have a
                    // null task wrapper pointer here.
                    if (data_ != nullptr)
                    {
                        hpx::util::external_timer::stop(data_);
                    }
                }
            }

            void yield()
            {
                if (!stopped)
                {
                    stopped = true;
                    // APEX internal actions are not timed. Otherwise, we would
                    // end up with recursive timers. So it's possible to have a
                    // null task wrapper pointer here.
                    if (data_ != nullptr)
                    {
                        hpx::util::external_timer::yield(data_);
                    }
                }
            }

            bool stopped;
            std::shared_ptr<task_wrapper> data_;
        };
    }    // namespace external_timer

#else
    namespace external_timer {

        struct task_wrapper
        {
        };

        inline std::shared_ptr<task_wrapper> new_task(thread_description const&,
            std::uint32_t, threads::thread_id_type const&)
        {
            return nullptr;
        }

        inline std::shared_ptr<task_wrapper> update_task(
            std::shared_ptr<task_wrapper>, thread_description const&)
        {
            return nullptr;
        }

        struct scoped_timer
        {
            explicit scoped_timer(std::shared_ptr<task_wrapper>) {}
            ~scoped_timer() = default;

            void stop(void) {}
            void yield(void) {}
        };
    }    // namespace external_timer
#endif
}}    // namespace hpx::util
