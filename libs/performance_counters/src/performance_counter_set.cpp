//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/performance_counter.hpp>
#include <hpx/performance_counters/performance_counter_set.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters {
    performance_counter_set::performance_counter_set(
        std::string const& name, bool reset)
      : invocation_count_(0)
      , print_counters_locally_(false)
    {
        add_counters(name, reset);
    }

    performance_counter_set::performance_counter_set(
        std::vector<std::string> const& names, bool reset)
      : invocation_count_(0)
      , print_counters_locally_(false)
    {
        add_counters(names, reset);
    }

    void performance_counter_set::release()
    {
        std::vector<hpx::id_type> ids;

        {
            std::lock_guard<mutex_type> l(mtx_);
            infos_.clear();
            std::swap(ids_, ids);
        }
    }

    std::size_t performance_counter_set::size() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return ids_.size();
    }

    std::vector<counter_info> performance_counter_set::get_counter_infos() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return infos_;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool performance_counter_set::find_counter(
        counter_info const& info, bool reset, error_code& ec)
    {
        // keep only local counters if requested
        if (print_counters_locally_)
        {
            counter_path_elements p;
            get_counter_path_elements(info.fullname_, p, ec);
            if (ec)
                return false;

            if (p.parentinstanceindex_ != hpx::get_locality_id())
            {
                if (&ec != &throws)
                    ec = make_success_code();
                return true;
            }
        }

        naming::id_type id = get_counter(info.fullname_, ec);
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "performance_counter_set::find_counter",
                hpx::util::format("unknown performance counter: '{1}' ({2})",
                    info.fullname_, ec.get_message()));
            return false;
        }

        {
            std::unique_lock<mutex_type> l(mtx_);
            infos_.push_back(info);
            ids_.push_back(id);
            reset_.push_back(reset ? 1 : 0);
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    void performance_counter_set::add_counters(
        std::string const& name, bool reset, error_code& ec)
    {
        using util::placeholders::_1;
        using util::placeholders::_2;

        discover_counter_func func = util::bind(
            &performance_counter_set::find_counter, this, _1, reset, _2);

        // do INI expansion on counter name
        std::string n(name);
        util::expand(n);

        // find matching counter types
        discover_counter_type(n, std::move(func), discover_counters_full, ec);
        if (ec)
            return;

        HPX_ASSERT(ids_.size() == infos_.size());
    }

    void performance_counter_set::add_counters(
        std::vector<std::string> const& names, bool reset, error_code& ec)
    {
        using util::placeholders::_1;
        using util::placeholders::_2;

        discover_counter_func func = util::bind(
            &performance_counter_set::find_counter, this, _1, reset, _2);

        for (std::string const& name : names)
        {
            // do INI expansion on counter name
            std::string n(name);
            util::expand(n);

            // find matching counter types
            discover_counter_type(n, func, discover_counters_full, ec);
            if (ec)
                return;
        }

        HPX_ASSERT(ids_.size() == infos_.size());
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<bool>> performance_counter_set::start()
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
        }

        std::vector<hpx::future<bool>> v;
        v.reserve(ids.size());

        // start all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.start());
        }

        return v;
    }

    bool performance_counter_set::start(launch::sync_policy, error_code& ec)
    {
        try
        {
            auto v = hpx::util::unwrap(start());
            return std::all_of(
                v.begin(), v.end(), [](bool val) { return val; });
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(ec, e, "performance_counter_set::start");
            return false;
        }
    }

    std::vector<hpx::future<bool>> performance_counter_set::stop()
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
        }

        std::vector<hpx::future<bool>> v;
        v.reserve(ids.size());

        // stop all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.stop());
        }

        return v;
    }

    bool performance_counter_set::stop(launch::sync_policy, error_code& ec)
    {
        try
        {
            auto v = hpx::util::unwrap(stop());
            return std::all_of(
                v.begin(), v.end(), [](bool val) { return val; });
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(ec, e, "performance_counter_set::stop");
            return false;
        }
    }

    std::vector<hpx::future<void>> performance_counter_set::reset()
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
        }

        std::vector<hpx::future<void>> v;
        v.reserve(ids.size());

        // reset all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.reset());
        }

        return v;
    }

    void performance_counter_set::reset(launch::sync_policy, error_code& ec)
    {
        try
        {
            hpx::util::unwrap(reset());
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(ec, e, "performance_counter_set::reset");
        }
    }

    std::vector<hpx::future<void>> performance_counter_set::reinit(bool reset)
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
        }

        std::vector<hpx::future<void>> v;
        v.reserve(ids.size());

        // re-initialize all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.reinit(reset));
        }

        return v;
    }

    void performance_counter_set::reinit(
        launch::sync_policy, bool reset, error_code& ec)
    {
        try
        {
            hpx::util::unwrap(reinit(reset));
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(ec, e, "performance_counter_set::reinit");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<counter_value>>
    performance_counter_set::get_counter_values(bool reset) const
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
            ++invocation_count_;
        }

        std::vector<hpx::future<counter_value>> v;
        v.reserve(ids.size());

        // reset all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            if (infos_[i].type_ == counter_histogram ||
                infos_[i].type_ == counter_raw_values)
            {
                continue;
            }

            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.get_counter_value(reset || reset_[i]));
        }

        return v;
    }

    std::vector<counter_value> performance_counter_set::get_counter_values(
        launch::sync_policy, bool reset, error_code& ec) const
    {
        try
        {
            return hpx::util::unwrap(get_counter_values(reset));
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(
                ec, e, "performance_counter_set::get_counter_values");
            return std::vector<counter_value>();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::vector<hpx::future<counter_values_array>>
    performance_counter_set::get_counter_values_array(bool reset) const
    {
        std::vector<hpx::id_type> ids;

        {
            std::unique_lock<mutex_type> l(mtx_);
            ids = ids_;
            ++invocation_count_;
        }

        std::vector<hpx::future<counter_values_array>> v;
        v.reserve(ids.size());

        // reset all performance counters
        for (std::size_t i = 0; i != ids.size(); ++i)
        {
            if (infos_[i].type_ != counter_histogram &&
                infos_[i].type_ != counter_raw_values)
            {
                continue;
            }

            performance_counters::performance_counter c(ids[i]);
            v.push_back(c.get_counter_values_array(reset || reset_[i]));
        }

        return v;
    }

    std::vector<counter_values_array>
    performance_counter_set::get_counter_values_array(
        launch::sync_policy, bool reset, error_code& ec) const
    {
        try
        {
            return hpx::util::unwrap(get_counter_values_array(reset));
        }
        catch (hpx::exception const& e)
        {
            HPX_RETHROWS_IF(
                ec, e, "performance_counter_set::get_counter_values_aray");
            return std::vector<counter_values_array>();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t performance_counter_set::get_invocation_count() const
    {
        std::unique_lock<mutex_type> l(mtx_);
        return invocation_count_;
    }
}}    // namespace hpx::performance_counters
