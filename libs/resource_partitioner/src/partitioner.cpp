//  Copyright (c)      2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/resource_partitioner/detail/partitioner.hpp>
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/topology/cpu_mask.hpp>

#include <hpx/modules/program_options.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace resource {
    ///////////////////////////////////////////////////////////////////////////
    std::vector<pu> pu::pus_sharing_core()
    {
        std::vector<pu> result;
        result.reserve(core_->pus_.size());

        for (pu const& p : core_->pus_)
        {
            if (p.id_ != id_)
            {
                result.push_back(p);
            }
        }
        return result;
    }

    std::vector<pu> pu::pus_sharing_numa_domain()
    {
        std::vector<pu> result;
        result.reserve(core_->domain_->cores_.size());

        for (core const& c : core_->domain_->cores_)
        {
            for (pu const& p : c.pus_)
            {
                if (p.id_ != id_)
                {
                    result.push_back(p);
                }
            }
        }
        return result;
    }

    std::vector<core> core::cores_sharing_numa_domain()
    {
        std::vector<core> result;
        result.reserve(domain_->cores_.size());

        for (core const& c : domain_->cores_)
        {
            if (c.id_ != id_)
            {
                result.push_back(c);
            }
        }
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        std::recursive_mutex& partitioner_mtx()
        {
            static std::recursive_mutex mtx;
            return mtx;
        }

        std::unique_ptr<detail::partitioner>& partitioner_ref()
        {
            static std::unique_ptr<detail::partitioner> part;
            return part;
        }

        std::unique_ptr<detail::partitioner>& get_partitioner()
        {
            std::lock_guard<std::recursive_mutex> l(partitioner_mtx());
            std::unique_ptr<detail::partitioner>& part = partitioner_ref();
            if (!part)
                part.reset(new detail::partitioner);
            return part;
        }

        void delete_partitioner()
        {
            // don't lock the mutex as otherwise will be still locked while
            // being destroyed (leading to problems on some platforms)
            std::unique_ptr<detail::partitioner>& part = partitioner_ref();
            if (part)
                part.reset();
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    struct partitioner_tag
    {
    };

    detail::partitioner& get_partitioner()
    {
        std::unique_ptr<detail::partitioner>& rp = detail::get_partitioner();

        if (!rp)
        {
            // if the resource partitioner is not accessed for the first time
            // if the command-line parsing has not yet been done
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::resource::get_partitioner",
                "can be called only after the resource partitioner has "
                "been initialized and before it has been deleted.");
        }

        if (!rp->cmd_line_parsed())
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hpx::resource::get_partitioner",
                "can be called only after the resource partitioner has "
                "been allowed to parse the command line options.");
        }

        return *rp;
    }

    bool is_partitioner_valid()
    {
        return detail::partitioner_ref() != nullptr;
    }

    namespace detail {
        detail::partitioner& create_partitioner(
            util::function_nonser<int(
                hpx::program_options::variables_map& vm)> const& f,
            hpx::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string> ini_config,
            resource::partitioner_mode rpmode, runtime_mode mode, bool check,
            std::vector<std::shared_ptr<components::component_registry_base>>&
                component_registries,
            int* result)
        {
            std::unique_ptr<detail::partitioner>& rp =
                detail::get_partitioner();

            if (rp->cmd_line_parsed())
            {
                if (check)
                {
                    // if the resource partitioner is not accessed for the
                    // first time if the command-line parsing has not yet
                    // been done
                    HPX_THROW_EXCEPTION(invalid_status,
                        "hpx::resource::get_partitioner",
                        "can be called only after the resource partitioner "
                        "has been allowed to parse the command line "
                        "options.");
                }

                // no need to parse a second time
                if (result != nullptr)
                {
                    *result = 0;
                }
            }
            else
            {
                int r = rp->parse(f, desc_cmdline, argc, argv,
                    std::move(ini_config), rpmode, mode, component_registries,
                    true);
                if (result != nullptr)
                {
                    *result = r;
                }
            }
            return *rp;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void partitioner::create_thread_pool(std::string const& name,
        scheduling_policy sched /*= scheduling_policy::unspecified*/,
        hpx::threads::policies::scheduler_mode mode)
    {
        partitioner_.create_thread_pool(name, sched, mode);
    }

    void partitioner::create_thread_pool(
        std::string const& name, scheduler_function scheduler_creation)
    {
        partitioner_.create_thread_pool(name, scheduler_creation);
    }

    void partitioner::set_default_pool_name(std::string const& name)
    {
        partitioner_.set_default_pool_name(name);
    }

    const std::string& partitioner::get_default_pool_name() const
    {
        return partitioner_.get_default_pool_name();
    }

    void partitioner::add_resource(pu const& p, std::string const& pool_name,
        bool exclusive, std::size_t num_threads /*= 1*/)
    {
        partitioner_.add_resource(p, pool_name, exclusive, num_threads);
    }

    void partitioner::add_resource(std::vector<pu> const& pv,
        std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(pv, pool_name, exclusive);
    }

    void partitioner::add_resource(
        core const& c, std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(c, pool_name, exclusive);
    }

    void partitioner::add_resource(std::vector<core>& cv,
        std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(cv, pool_name, exclusive);
    }

    void partitioner::add_resource(numa_domain const& nd,
        std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(nd, pool_name, exclusive);
    }

    void partitioner::add_resource(std::vector<numa_domain> const& ndv,
        std::string const& pool_name, bool exclusive /*= true*/)
    {
        partitioner_.add_resource(ndv, pool_name, exclusive);
    }

    std::vector<numa_domain> const& partitioner::numa_domains() const
    {
        return partitioner_.numa_domains();
    }

    hpx::threads::topology const& partitioner::get_topology() const
    {
        return partitioner_.get_topology();
    }

    std::size_t partitioner::get_number_requested_threads()
    {
        return partitioner_.threads_needed();
    }

    util::command_line_handling& partitioner::get_command_line_switches()
    {
        return partitioner_.get_command_line_switches();
    }

    // Does initialization of all resources and internal data of the
    // resource partitioner called in hpx_init
    void partitioner::configure_pools()
    {
        partitioner_.configure_pools();
    }

    // Return the initialization result for this resource_partitioner
    int partitioner::parse_result() const
    {
        return partitioner_.parse_result();
    }

}}    // namespace hpx::resource
