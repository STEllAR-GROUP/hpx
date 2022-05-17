// Copyright (c) 2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/asssert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/components/performance_counters/power/power_counter.hpp>

#include <cstdint>
#include <mutex>
#include <string>

#include <pwr.h>

namespace hpx::performance_counters::power {

    class power
    {
    public:
        power()
        {
            hpx::register_startup_function([this]() { this->init(); });
        }

        ~power()
        {
            if (cntxt != nullptr)
            {
                PWR_CntxtDestroy(cntxt);
            }
        }

        power(power const&) = delete;
        power(power&&) = delete;
        power& operator=(power const&) = delete;
        power& operator=(power&&) = delete;

        std::uint64_t get_value(bool reset)
        {
            std::lock_guard l(mtx);

            PWR_Time now = 0;
            double energy = 0.0;
            sample_power(energy, now);

            std::uint64_t value = std::uint64_t(
                (energy - base_energy) / ((now - base_time) / 1000000000.0));

            if (reset)
            {
                base_energy = energy;
                base_time = now;
            }

            return value;
        }

    protected:
        // delayed initialization
        void init()
        {
            std::lock_guard l(mtx);

            // Get a context
            int rc = PWR_CntxtInit(
                PWR_CNTXT_DEFAULT, PWR_ROLE_APP, "application", &cntxt);
            if (rc != PWR_RET_SUCCESS || cntxt == nullptr)
            {
                HPX_THROW_EXCEPTION(bad_request, "power::init",
                    hpx::format(
                        "PWR_CntxtInit returned error: {} (locality: {})", rc,
                        hpx::get_locality_name()));
            }

            rc = PWR_CntxtGetObjByName(cntxt, "plat.node", &obj);
            if (rc != PWR_RET_SUCCESS || obj == nullptr)
            {
                HPX_THROW_EXCEPTION(bad_request, "power::init",
                    hpx::format("PWR_CntxtGetObjByName returned error: {} "
                                "(locality: {})",
                        rc, hpx::get_locality_name()));
            }

            sample_power(base_energy, base_time);
        }

        void sample_power(double& energy, PWR_Time& pwr_time)
        {
            rc = PWR_ObjAttrGetValue(obj, PWR_ATTR_ENERGY, &energy, &pwr_time);
            if (rc != PWR_RET_SUCCESS)
            {
                HPX_THROW_EXCEPTION(bad_request, "power::sample_power",
                    hpx::format(
                        "PWR_ObjAttrGetValue returned error: {} (locality: {})",
                        rc, hpx::get_locality_name()));
            }
            return energy;
        }

    private:
        hpx::spinlock mtx;
        PWR_Cntxt cntxt = nullptr;
        PWR_Obj obj = nullptr;
        double base_energy = 0.0;
        PWR_Time base_time = 0;
    };

    power power_api;

    ///////////////////////////////////////////////////////////////////////////
    // returns overall power consumption
    std::uint64_t average_power_consumption(bool reset)
    {
        return power_api.get_value(reset);
    }
}    // namespace hpx::performance_counters::power
