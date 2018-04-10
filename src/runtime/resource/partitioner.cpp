//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/compat/mutex.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/resource/partitioner.hpp>
#include <hpx/runtime/thread_pool_helpers.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>

#include <boost/program_options.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace resource
{
    ///////////////////////////////////////////////////////////////////////////
    std::size_t get_num_thread_pools()
    {
        return get_partitioner().get_num_pools();
    }

    std::size_t get_num_threads()
    {
        return get_partitioner().get_num_threads();
    }

    std::size_t get_num_threads(std::string const& pool_name)
    {
        return get_partitioner().get_num_threads(pool_name);
    }

    std::size_t get_num_threads(std::size_t pool_index)
    {
        return get_partitioner().get_num_threads(pool_index);
    }

    std::size_t get_pool_index(std::string const& pool_name)
    {
        return get_partitioner().get_pool_index(pool_name);
    }

    std::string const& get_pool_name(std::size_t pool_index)
    {
        return get_partitioner().get_pool_name(pool_index);
    }

    threads::thread_pool_base& get_thread_pool(
        std::string const& pool_name)
    {
        return get_runtime().get_thread_manager().get_pool(pool_name);
    }

    threads::thread_pool_base& get_thread_pool(std::size_t pool_index)
    {
        return get_thread_pool(get_pool_name(pool_index));
    }

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
    namespace detail
    {
        compat::mutex& partitioner_mtx()
        {
            static compat::mutex mtx;
            return mtx;
        }

        std::unique_ptr<detail::partitioner>& partitioner_ref()
        {
            static std::unique_ptr<detail::partitioner> part;
            return part;
        }

        std::unique_ptr<detail::partitioner>& get_partitioner()
        {
            std::lock_guard<compat::mutex> l(partitioner_mtx());
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
    }

    ///////////////////////////////////////////////////////////////////////////
    struct partitioner_tag {};

    detail::partitioner &get_partitioner()
    {
        std::unique_ptr<detail::partitioner>& rp = detail::get_partitioner();

        if (!rp)
        {
            // if the resource partitioner is not accessed for the first time
            // if the command-line parsing has not yet been done
            throw std::invalid_argument(
                "hpx::resource::get_partitioner() can be called only after "
                "the resource partitioner has been initialized and before it "
                "has been deleted");
        }

        if (!rp->cmd_line_parsed())
        {
            if (get_runtime_ptr() != nullptr)
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "hpx::resource::get_partitioner",
                    "can be called only after the resource partitioner has "
                    "been allowed to parse the command line options.");
            }
            else
            {
                // if the resource partitioner is not accessed for the first time
                // if the command-line parsing has not yet been done
                throw std::invalid_argument(
                    "hpx::resource::get_partitioner() can be called only after "
                    "the resource partitioner has been allowed to parse the "
                    "command line options.");
            }
        }

        return *rp;
    }

    bool is_partitioner_valid()
    {
        return bool(detail::get_partitioner());
    }

    namespace detail
    {
        detail::partitioner &create_partitioner(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description const& desc_cmdline,
            int argc, char **argv, std::vector<std::string> ini_config,
            resource::partitioner_mode rpmode, runtime_mode mode,
            bool check)
        {
            std::unique_ptr<detail::partitioner>& rp = detail::get_partitioner();

            if (rp->cmd_line_parsed())
            {
                if (check)
                {
                    // if the resource partitioner is not accessed for the
                    // first time if the command-line parsing has not yet
                    // been done
                    if (get_runtime_ptr() != nullptr)
                    {
                        HPX_THROW_EXCEPTION(invalid_status,
                            "hpx::resource::get_partitioner",
                            "can be called only after the resource partitioner "
                            "has been allowed to parse the command line "
                            "options.");
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "hpx::resource::get_partitioner() can be called "
                            "only after the resource partitioner has been "
                            "allowed to parse the command line options.");
                    }
                }
                // no need to parse a second time
            }
            else
            {
                rp->parse(f, desc_cmdline, argc, argv, std::move(ini_config),
                    rpmode, mode);
            }
            return *rp;
        }
    }

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

    const std::string & partitioner::get_default_pool_name() const {
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
}}    // namespace hpx
