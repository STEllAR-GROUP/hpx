//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once // prevent multiple inclusions of this header file.

#include <hpx/config.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/util/thread_description.hpp>
#include <hpx/runtime/get_num_localities.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <mutex>

#ifdef HPX_HAVE_APEX
#include "apex_api.hpp"
#include "apex_policies.hpp"
#endif

namespace hpx { namespace util
{
#ifdef HPX_HAVE_APEX

#ifdef HPX_HAVE_PARCEL_COALESCING
/* these are related to parcel coalescing policy
 */
    struct apex_parcel_coalescing_policy {
        /* pointer to the policy (so we can stop it) */
        apex_policy_handle * policy_handle;
        /* The tuning request */
        apex_tuning_request * request;
        int tuning_window;
        std::string counter_name;
        std::string name;
        static apex_parcel_coalescing_policy * instance;
        static std::mutex params_mutex;

        void set_coalescing_params() {
            std::cerr << __func__ << ": Setting params!" << std::endl;
            std::shared_ptr<apex_param_long> parcel_count_param = 
                std::static_pointer_cast<apex_param_long>(request->get_param("parcel_count"));
            std::shared_ptr<apex_param_long> buffer_time_param = 
                std::static_pointer_cast<apex_param_long>(request->get_param("buffer_time"));
            const int parcel_count = parcel_count_param->get_value();
            const int buffer_time = buffer_time_param->get_value();
            std::cerr << "setting parcel count: " << parcel_count << std::endl;
            hpx::set_config_entry(
                "hpx.plugins.coalescing_message_handler.num_messages",
                parcel_count);
            std::cerr << "setting buffer time: " << buffer_time << std::endl;
            hpx::set_config_entry(
                "hpx.plugins.coalescing_message_handler.interval",
                buffer_time);
            std::cerr << "done." << std::endl;
        }

        /* policy function */
        static int policy(const apex_context context) {
            //std::cerr << "policy" << std::endl;
            apex_profile * profile = apex::get_profile(instance->counter_name);
            if(profile != nullptr && profile->calls >= instance->tuning_window) {
                std::cerr << "profile calls: " << profile->calls << " " << std::endl;
                //std::cout << __func__ << ": Got results!" << std::endl;
                // Evaluate the results
                apex::custom_event(instance->request->get_trigger(), NULL);
                // get new settings from the search
                instance->set_coalescing_params();
                // Reset counter so each measurement is fresh.
                apex::reset(instance->counter_name);
            }
            return APEX_NOERROR;
        }   

        /* Constructor */
        apex_parcel_coalescing_policy() : tuning_window(10),
            name("HPX parcel coalescing")
        {
            //std::cout << __func__ << ": Constructor!" << std::endl;
            std::stringstream ss;
            ///threads{locality#0/total}/time/average-overhead
            ss << "/threads{locality#" << hpx::get_locality_id();
            ss << "/total}/time/average-overhead";
            counter_name = std::string(ss.str());
            /* Create a metric to be queried for this policy */
            std::function<double(void)> metric = [=]()->double{
                apex_profile * profile = apex::get_profile(counter_name);
                if (profile == nullptr || profile->calls == 0) {
                    // no data yet
                    return 0.0;
                }
                double result = profile->accumulated/profile->calls;
                return result;
            };
            request = new apex_tuning_request(name);
            request->set_metric(metric);
            request->set_strategy(apex_ah_tuning_strategy::NELDER_MEAD);
            request->add_param_long("parcel_count", 50, 1, 256, 1);
            request->add_param_long("buffer_time", 100, 1, 7000, 1);
            request->set_trigger(apex::register_custom_event(name));
            /* add the request */
            apex::setup_custom_tuning(*request);

            /* register the tuning policy */
            //std::function<int(apex_context const&)> policy_fn{policy};
            policy_handle = apex::register_periodic_policy(500000, policy);
            if(policy_handle == nullptr) {
                std::cerr << "Error registering policy!" << std::endl;
            }
        }
        static void initialize() {
            if (instance == nullptr) {
                instance = new apex_parcel_coalescing_policy();
            }
        }
        static void finalize() {
            //std::cout << __func__ << ": Destructor!" << std::endl;
            if (instance != nullptr) {
                delete instance;
                instance = nullptr;
            }
        }
    };
#endif

    static void hpx_util_apex_init_startup(void)
    {
        apex::init(nullptr, hpx::get_locality_id(),
            hpx::get_initial_num_localities());
#ifdef HPX_HAVE_PARCEL_COALESCING
        hpx::register_startup_function([&](){
            apex_parcel_coalescing_policy::initialize();
        });
#endif
    }

    inline void apex_init()
    {
        hpx_util_apex_init_startup();
        //hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
    }

    inline void apex_finalize()
    {
#ifdef HPX_HAVE_PARCEL_COALESCING
        apex_parcel_coalescing_policy::finalize();
#endif
        apex::finalize();
    }

    struct apex_wrapper
    {
        apex_wrapper(thread_description const& name)
          : name_(name), stopped(false)
        {
            if (name_.kind() == thread_description::data_type_description)
            {
                profiler_ = apex::start(name_.get_description());
            }
            else
            {
                profiler_ = apex::start(
                    apex_function_address(name_.get_address()));
            }
        }
        apex_wrapper(thread_description const& name, uint64_t id)
          : name_(name), stopped(false)
        {
            if (name_.kind() == thread_description::data_type_description)
            {
                profiler_ = apex::start(name_.get_description(), id);
            }
            else
            {
                profiler_ = apex::start(
                    apex_function_address(name_.get_address()), id);
            }
        }
        ~apex_wrapper()
        {
            stop();
        }

        void stop() {
            if(!stopped) {
                stopped = true;
                apex::stop(profiler_);
            }
        }

        void yield() {
            if(!stopped) {
                stopped = true;
                apex::yield(profiler_);
            }
        }

        thread_description name_;
        bool stopped;
        apex::profiler * profiler_;
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv)
        {
            //apex::init(nullptr, hpx::get_locality_id(),
            //    hpx::get_initial_num_localities());
            hpx::register_pre_startup_function(&hpx_util_apex_init_startup);
        }
        ~apex_wrapper_init()
        {
//#ifdef HPX_HAVE_PARCEL_COALESCING
            apex_parcel_coalescing_policy::finalize();
//#endif
            apex::finalize();
        }
    };

#else
    inline void apex_init() {}
    inline void apex_finalize() {}

    struct apex_wrapper
    {
        apex_wrapper(thread_description const& name) {}
        ~apex_wrapper() {}
    };

    struct apex_wrapper_init
    {
        apex_wrapper_init(int argc, char **argv) {}
        ~apex_wrapper_init() {}
    };
#endif
}}

