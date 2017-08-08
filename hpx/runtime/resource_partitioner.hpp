//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER)
#define HPX_RESOURCE_PARTITIONER

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology_info.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/util/tuple.hpp>
//
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>

#include <boost/atomic.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if !defined(HPX_EXPORTS)
// This function must be implemented by the application.
int hpx_main(boost::program_options::variables_map& vm);
#endif

namespace hpx {

namespace resource
{
    class resource_partitioner;

    // resource_partitioner mode
    enum resource_partitioner_mode
    {
        mode_default = 0,
        mode_allow_oversubscription = 1,
        mode_allow_dynamic_pools = 2
    };
}

// if the resource partitioner is accessed before the HPX runtime has started
// then on first access, this function should be used, to ensure that command line
// affinity binding options are honored. Use this function signature only once
// and thereafter use the parameter free version.
HPX_EXPORT resource::resource_partitioner& get_resource_partitioner(
    util::function_nonser<
        int(boost::program_options::variables_map& vm)
    > const& f,
    boost::program_options::options_description const& desc_cmdline,
    int argc, char** argv, std::vector<std::string> ini_config,
    resource::resource_partitioner_mode rpmode = resource::mode_default,
    runtime_mode mode = runtime_mode_default,
    bool check = true);

#if !defined(HPX_EXPORTS)
typedef int (*hpx_main_type)(boost::program_options::variables_map&);

inline resource::resource_partitioner& get_resource_partitioner(
    int argc, char** argv,
    resource::resource_partitioner_mode rpmode = resource::mode_default,
    runtime_mode mode = runtime_mode_default, bool check = true)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(),
        rpmode, mode, check);
}

inline resource::resource_partitioner &get_resource_partitioner(
    int argc, char **argv, std::vector<std::string> ini_config,
    resource::resource_partitioner_mode rpmode = resource::mode_default,
    runtime_mode mode = runtime_mode_default, bool check = true)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::move(ini_config),
        rpmode, mode, check);
}

///////////////////////////////////////////////////////////////////////////////
inline resource::resource_partitioner &get_resource_partitioner(
    boost::program_options::options_description const& desc_cmdline,
    int argc, char **argv,
    resource::resource_partitioner_mode rpmode = resource::mode_default,
    runtime_mode mode = runtime_mode_default, bool check = true)
{
    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(),
        rpmode, mode, check);
}

inline resource::resource_partitioner &get_resource_partitioner(
    boost::program_options::options_description const& desc_cmdline,
    int argc, char **argv, std::vector<std::string> ini_config,
    resource::resource_partitioner_mode rpmode = resource::mode_default,
    runtime_mode mode = runtime_mode_default, bool check = true)
{
    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, ini_config, rpmode, mode, check);
}

#endif

///////////////////////////////////////////////////////////////////////////////
// May be used anywhere in code and returns a reference to the single, global
// resource partitioner
HPX_EXPORT hpx::resource::resource_partitioner& get_resource_partitioner();

namespace resource {

    struct core;
    struct numa_domain;

    struct pu
    {
    private:
        std::size_t id_;
        core *core_;
        std::vector<pu> pus_sharing_core();
        std::vector<pu> pus_sharing_numa_domain();
        std::size_t thread_occupancy_;
        // indicates the number of threads that should run on this PU
        //  0: this PU is not exposed by the affinity bindings
        //  1: normal occupancy
        // >1: oversubscription
        mutable std::size_t thread_occupancy_count_;
        // counts number of threads bound to this PU
        friend struct core;
        friend struct numa_domain;
        friend class resource_partitioner;

    public:
        std::size_t id() const
        {
            return id_;
        }
    };

    struct core
    {
    private:
        std::size_t id_;
        numa_domain *domain_;
        std::vector<pu> pus_;
        std::vector<core> cores_sharing_numa_domain();
        friend struct pu;
        friend struct numa_domain;
        friend class resource_partitioner;

    public:
        const std::vector<pu> &pus() const
        {
            return pus_;
        }
        std::size_t id() const
        {
            return id_;
        }
    };

    struct numa_domain
    {
    private:
        std::size_t id_;
        std::vector<core> cores_;
        friend struct pu;
        friend struct core;
        friend class resource_partitioner;

    public:
        const std::vector<core> &cores() const
        {
            return cores_;
        }
        std::size_t id() const
        {
            return id_;
        }
    };

    using scheduler_function =
        util::function_nonser<
            std::unique_ptr<hpx::threads::detail::thread_pool_base>(
                hpx::threads::policies::callback_notifier&,
                std::size_t, std::size_t, std::size_t, std::string const&
            )>;

    // scheduler assigned to thread_pool
    // choose same names as in command-line options except with _ instead of -
    enum scheduling_policy
    {
        user_defined = -2,
        unspecified = -1,
        local = 0,
        local_priority_fifo = 1,
        local_priority_lifo = 2,
        static_ = 3,
        static_priority = 4,
        abp_priority = 5,
        hierarchy = 6,
        periodic_priority = 7,
        throttle = 8
    };

    class HPX_EXPORT resource_partitioner;

    namespace detail
    {
        // structure used to encapsulate all characteristics of thread_pools
        // as specified by the user in int main()
        class init_pool_data
        {
        public:
            // mechanism for adding resources (zero-based index)
            void add_resource(std::size_t pu_index, bool exclusive,
                std::size_t num_threads);

            void print_pool(std::ostream&) const;

            void assign_pu(std::size_t virt_core);
            void unassign_pu(std::size_t virt_core);

            bool pu_is_exclusive(std::size_t virt_core) const;
            bool pu_is_assigned(std::size_t virt_core) const;

            friend class resource::resource_partitioner;

            // counter ... overall, in all the thread pools
            static std::size_t num_threads_overall;

        private:
            init_pool_data(const std::string &name,
                scheduling_policy = scheduling_policy::unspecified);

            init_pool_data(std::string const& name, scheduler_function create_func);

            std::string pool_name_;
            scheduling_policy scheduling_policy_;

            // PUs this pool is allowed to run on
            std::vector<threads::mask_type> assigned_pus_;  // mask

            // pu index/exclusive/assigned
            std::vector<util::tuple<std::size_t, bool, bool>> assigned_pu_nums_;

            // counter for number of threads bound to this pool
            std::size_t num_threads_;
            scheduler_function create_function_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class resource_partitioner
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        // constructor: users shouldn't use the constructor
        // but rather get_resource_partitioner
        resource_partitioner();

        void print_init_pool_data(std::ostream&) const;

        // create a thread_pool
        void create_thread_pool(std::string const& name,
            scheduling_policy sched = scheduling_policy::unspecified);

        // create a thread_pool with a callback function for creating a custom
        // scheduler
        void create_thread_pool(std::string const& name,
            scheduler_function scheduler_creation);

        // Functions to add processing units to thread pools via
        // the pu/core/numa_domain API
        void add_resource(hpx::resource::pu const& p,
            std::string const& pool_name, std::size_t num_threads = 1)
        {
            add_resource(p, pool_name, true, num_threads);
        }
        void add_resource(hpx::resource::pu const& p,
            std::string const& pool_name, bool exclusive,
            std::size_t num_threads = 1);
        void add_resource(const std::vector<hpx::resource::pu>& pv,
            std::string const& pool_name, bool exclusive = true);
        void add_resource(const hpx::resource::core& c,
            std::string const& pool_name, bool exclusive = true);
        void add_resource(const std::vector<hpx::resource::core>& cv,
            std::string const& pool_name, bool exclusive = true);
        void add_resource(const hpx::resource::numa_domain& nd,
            std::string const& pool_name, bool exclusive = true);
        void add_resource(const std::vector<hpx::resource::numa_domain>& ndv,
            std::string const& pool_name, bool exclusive = true);

        // called by constructor of scheduler_base
        threads::policies::detail::affinity_data const& get_affinity_data() const
        {
            return affinity_data_;
        }

        // Does initialization of all resources and internal data of the
        // resource partitioner called in hpx_init
        void configure_pools();

        ////////////////////////////////////////////////////////////////////////
        scheduling_policy which_scheduler(std::string const& pool_name);
        threads::topology &get_topology() const;
        util::command_line_handling &get_command_line_switches();

        std::size_t get_num_distinct_pus() const;

        std::size_t get_num_pools() const;

        std::size_t get_num_threads() const;
        std::size_t get_num_threads(std::string const& pool_name) const;
        std::size_t get_num_threads(std::size_t pool_index) const;

        std::string const& get_pool_name(std::size_t index) const;
        std::size_t get_pool_index(std::string const& pool_name) const;

        std::size_t get_pu_num(std::size_t global_thread_num);
        threads::mask_cref_type get_pu_mask(std::size_t global_thread_num) const;

        bool cmd_line_parsed() const;
        int parse(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, std::vector<std::string> ini_config,
            resource::resource_partitioner_mode rpmode,
            runtime_mode mode, bool fill_internal_topology = true);

        scheduler_function get_pool_creator(size_t index) const;

        std::vector<numa_domain> const& numa_domains() const
        {
            return numa_domains_;
        }

        bool terminate_after_parse()
        {
            return cfg_.parse_terminate_;
        }

        std::size_t cores_needed() const
        {
            // should have been initialized by now
            HPX_ASSERT(cores_needed_ != std::size_t(-1));
            return cores_needed_;
        }

        // manage dynamic footprint of pools
        void assign_pu(std::string const& pool_name, std::size_t virt_core);
        void unassign_pu(std::string const& pool_name, std::size_t virt_core);

        std::size_t shrink_pool(std::string const& pool_name,
            util::function_nonser<void(std::size_t)> const& remove_pu);
        std::size_t expand_pool(std::string const& pool_name,
            util::function_nonser<void(std::size_t)> const& add_pu);

    private:
        ////////////////////////////////////////////////////////////////////////
        void fill_topology_vectors();
        bool pu_exposed(std::size_t pid);

        ////////////////////////////////////////////////////////////////////////
        // called in hpx_init run_or_start
        void set_init_affinity_data();
        void setup_pools();
        void setup_schedulers();
        void reconfigure_affinities();
        bool check_empty_pools() const;

        // helper functions
        detail::init_pool_data const& get_pool_data(
            std::size_t pool_index) const;

        // has to be private because pointers become invalid after data member
        // thread_pools_ is resized we don't want to allow the user to use it
        detail::init_pool_data const& get_pool_data(
            std::string const& pool_name) const;
        detail::init_pool_data& get_pool_data(std::string const& pool_name);

        void set_scheduler(scheduling_policy sched, std::string const& pool_name);

        ////////////////////////////////////////////////////////////////////////
        // counter for instance numbers
        static boost::atomic<int> instance_number_counter_;

        // holds all of the command line switches
        util::command_line_handling cfg_;
        std::size_t cores_needed_;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        mutable mutex_type mtx_;
        std::vector<detail::init_pool_data> initial_thread_pools_;

        // reference to the topology and affinity data
        threads::hwloc_topology_info &topology_;
        hpx::threads::policies::detail::affinity_data affinity_data_;

        // contains the internal topology back-end used to add resources to
        // initial_thread_pools
        std::vector<numa_domain> numa_domains_;

        // store policy flags determining the general behavior of the
        // resource_partitioner
        resource_partitioner_mode mode_;
    };

}    // namespace resource
}    // namespace hpx

#endif
