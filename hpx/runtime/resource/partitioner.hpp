//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER_AUG_10_2017_1005AM)
#define HPX_RESOURCE_PARTITIONER_AUG_10_2017_1005AM

#include <hpx/config.hpp>
#include <hpx/runtime/resource/partitioner_fwd.hpp>
#include <hpx/runtime/resource/detail/create_partitioner.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/util/function.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace resource
{
    ///////////////////////////////////////////////////////////////////////////
    class pu
    {
        HPX_CONSTEXPR static const std::size_t invalid_pu_id = std::size_t(-1);

    public:
        explicit pu(std::size_t id = invalid_pu_id, core *core = nullptr,
                std::size_t thread_occupancy = 0)
          : id_(id)
          , core_(core)
          , thread_occupancy_(thread_occupancy)
          , thread_occupancy_count_(0)
        {
        }

        std::size_t id() const
        {
            return id_;
        }

    private:
        friend class core;
        friend class numa_domain;
        friend class resource::detail::partitioner;

        std::vector<pu> pus_sharing_core();
        std::vector<pu> pus_sharing_numa_domain();

        std::size_t id_;
        core *core_;

        // indicates the number of threads that should run on this PU
        //  0: this PU is not exposed by the affinity bindings
        //  1: normal occupancy
        // >1: oversubscription
        std::size_t thread_occupancy_;

        // counts number of threads bound to this PU
        mutable std::size_t thread_occupancy_count_;
    };

    class core
    {
        HPX_CONSTEXPR static const std::size_t invalid_core_id = std::size_t(-1);

    public:
        explicit core(std::size_t id = invalid_core_id,
                numa_domain *domain = nullptr)
          : id_(id)
          , domain_(domain)
        {
        }

        std::vector<pu> const& pus() const
        {
            return pus_;
        }
        std::size_t id() const
        {
            return id_;
        }

    private:
        std::vector<core> cores_sharing_numa_domain();

        friend class pu;
        friend class numa_domain;
        friend class resource::detail::partitioner;

        std::size_t id_;
        numa_domain *domain_;
        std::vector<pu> pus_;
    };

    class numa_domain
    {
        HPX_CONSTEXPR static const std::size_t invalid_numa_domain_id =
            std::size_t(-1);

    public:
        explicit numa_domain(std::size_t id = invalid_numa_domain_id)
          : id_(id)
        {
        }

        std::vector<core> const& cores() const
        {
            return cores_;
        }
        std::size_t id() const
        {
            return id_;
        }

    private:
        friend class pu;
        friend class core;
        friend class resource::detail::partitioner;

        std::size_t id_;
        std::vector<core> cores_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class partitioner
    {
    public:
        partitioner(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string> ini_config,
            resource::partitioner_mode rpmode = resource::mode_default,
            runtime_mode mode = runtime_mode_default)
          : partitioner_(detail::create_partitioner(f, desc_cmdline, argc,
                argv, std::move(ini_config), rpmode, mode))
        {}

#if !defined(HPX_EXPORTS)
        partitioner(int argc, char** argv,
            resource::partitioner_mode rpmode = resource::mode_default,
            runtime_mode mode = runtime_mode_default)
          : partitioner_(
                detail::create_partitioner(argc, argv, rpmode, mode))
        {}

        partitioner(int argc, char** argv,
            std::vector<std::string> ini_config,
            resource::partitioner_mode rpmode = resource::mode_default,
            runtime_mode mode = runtime_mode_default)
          : partitioner_(detail::create_partitioner(
                argc, argv, std::move(ini_config), rpmode, mode))
        {}

        partitioner(
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv,
            resource::partitioner_mode rpmode = resource::mode_default,
            runtime_mode mode = runtime_mode_default)
          : partitioner_(detail::create_partitioner(
                desc_cmdline, argc, argv, rpmode, mode))
        {}

        partitioner(
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv, std::vector<std::string> ini_config,
            resource::partitioner_mode rpmode = resource::mode_default,
            runtime_mode mode = runtime_mode_default)
          : partitioner_(detail::create_partitioner(
                desc_cmdline, argc, argv, std::move(ini_config), rpmode, mode))
        {}
#endif

        ///////////////////////////////////////////////////////////////////////
        // Create one of the predefined thread pools
        HPX_EXPORT void create_thread_pool(std::string const& name,
            scheduling_policy sched = scheduling_policy::unspecified);

        // Create a custom thread pool with a callback function
        HPX_EXPORT void create_thread_pool(std::string const& name,
            scheduler_function scheduler_creation);

        ///////////////////////////////////////////////////////////////////////
        // Functions to add processing units to thread pools via
        // the pu/core/numa_domain API
        void add_resource(hpx::resource::pu const& p,
            std::string const& pool_name, std::size_t num_threads = 1)
        {
            add_resource(p, pool_name, true, num_threads);
        }
        HPX_EXPORT void add_resource(hpx::resource::pu const& p,
            std::string const& pool_name, bool exclusive,
            std::size_t num_threads = 1);
        HPX_EXPORT void add_resource(std::vector<hpx::resource::pu> const& pv,
            std::string const& pool_name, bool exclusive = true);
        HPX_EXPORT void add_resource(hpx::resource::core const& c,
            std::string const& pool_name, bool exclusive = true);
        HPX_EXPORT void add_resource(std::vector<hpx::resource::core>& cv,
            std::string const& pool_name, bool exclusive = true);
        HPX_EXPORT void add_resource(hpx::resource::numa_domain const& nd,
            std::string const& pool_name, bool exclusive = true);
        HPX_EXPORT void add_resource(
            std::vector<hpx::resource::numa_domain> const& ndv,
            std::string const& pool_name, bool exclusive = true);

        // Access all available NUMA domains
        HPX_EXPORT std::vector<numa_domain> const& numa_domains() const;

    private:
        detail::partitioner& partitioner_;
    };
}}

#endif
