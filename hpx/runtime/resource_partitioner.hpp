//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER)
#define HPX_RESOURCE_PARTITIONER

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

namespace resource {
    class resource_partitioner;
}

// if the resource partitioner is accessed before the HPX runtime has started
// then on first access, this function should be used, to ensure that command line
// affinity binding options are honored. Use this function signature only once
// and thereafter use the parameter free version.
HPX_EXPORT resource::resource_partitioner& get_resource_partitioner(
    util::function_nonser<
        int(boost::program_options::variables_map& vm)
    > const& f,
    boost::program_options::options_description const& desc_cmdline, int argc,
    char** argv, std::vector<std::string> ini_config, runtime_mode mode,
    bool check = true);

#if !defined(HPX_EXPORTS)
typedef int (*hpx_main_type)(boost::program_options::variables_map&);

inline resource::resource_partitioner& get_resource_partitioner(
    int argc, char** argv)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(),
        runtime_mode_default);
}

inline resource::resource_partitioner &get_resource_partitioner(
    boost::program_options::options_description const& desc_cmdline, int argc,
    char **argv, bool check = true)
{
    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(),
        runtime_mode_default, check);
}

inline resource::resource_partitioner &get_resource_partitioner(
    int argc, char **argv, std::vector<std::string> ini_config)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::move(ini_config), runtime_mode_default);
}

inline resource::resource_partitioner &get_resource_partitioner(
    int argc, char **argv, runtime_mode mode)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(0), mode);
}

inline resource::resource_partitioner &get_resource_partitioner(
    boost::program_options::options_description const& desc_cmdline, int argc,
    char **argv, std::vector<std::string> ini_config)
{
    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::move(ini_config), runtime_mode_default);
}

inline resource::resource_partitioner &get_resource_partitioner(
    boost::program_options::options_description const& desc_cmdline, int argc,
    char **argv, runtime_mode mode)
{
    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::vector<std::string>(), mode);
}

inline resource::resource_partitioner& get_resource_partitioner(int argc,
    char** argv, std::vector<std::string> ini_config, runtime_mode mode)
{
    boost::program_options::options_description desc_cmdline(
        std::string("Usage: ") + HPX_APPLICATION_STRING + " [options]");

    return get_resource_partitioner(static_cast<hpx_main_type>(::hpx_main),
        desc_cmdline, argc, argv, std::move(ini_config), mode);
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
            void add_resource(std::size_t pu_index, std::size_t num_threads);

            void print_pool(std::ostream&) const;

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
            std::vector<threads::mask_type> assigned_pus_;
            // counter for number of threads bound to this pool
            std::size_t num_threads_;
            scheduler_function create_function_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    class resource_partitioner
    {
    public:
        // constructor: users shouldn't use the constructor
        // but rather get_resource_partitioner
        resource_partitioner();

        void print_init_pool_data(std::ostream&) const;

        // create a thread_pool
        void create_thread_pool(const std::string &name,
            scheduling_policy sched = scheduling_policy::unspecified);

        // create a thread_pool with a callback function for creating a custom
        // scheduler
        void create_thread_pool(const std::string &name,
            scheduler_function scheduler_creation);

        // Functions to add processing units to thread pools via
        // the pu/core/numa_domain API
        void add_resource(hpx::resource::pu const& p,
            const std::string& pool_name, std::size_t num_threads = 1);
        void add_resource(const std::vector<hpx::resource::pu>& pv,
            const std::string& pool_name);
        void add_resource(const hpx::resource::core& c,
            const std::string& pool_name);
        void add_resource(const std::vector<hpx::resource::core>& cv,
            const std::string& pool_name);
        void add_resource(const hpx::resource::numa_domain& nd,
            const std::string& pool_name);
        void add_resource(const std::vector<hpx::resource::numa_domain>& ndv,
            const std::string& pool_name);

        // called by constructor of scheduler_base
        threads::policies::detail::affinity_data const& get_affinity_data() const
        {
            return affinity_data_;
        }
        threads::policies::detail::affinity_data& get_affinity_data()
        {
            return affinity_data_;
        }

        // Does initialization of all resources and internal data of the
        // resource partitioner called in hpx_init
        void configure_pools();

        ////////////////////////////////////////////////////////////////////////
        scheduling_policy which_scheduler(const std::string &pool_name);
        threads::topology &get_topology() const;
        util::command_line_handling &get_command_line_switches();

        std::size_t get_num_threads() const;
        std::size_t get_num_pools() const;
        std::size_t get_num_threads(std::string const& pool_name) const;
        std::size_t get_num_threads(std::size_t pool_index) const;

        detail::init_pool_data const& get_pool_data(std::size_t pool_index) const;

        std::string const& get_pool_name(std::size_t index) const;
        std::size_t get_pool_index(const std::string &pool_name) const;

        size_t get_pu_num(std::size_t global_thread_num);
        threads::mask_cref_type get_pu_mask(std::size_t global_thread_num) const;

        bool cmd_line_parsed() const;
        int parse(
            util::function_nonser<
                int(boost::program_options::variables_map& vm)
            > const& f,
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, std::vector<std::string> ini_config,
            runtime_mode mode, bool fill_internal_topology = true);

        scheduler_function const& get_pool_creator(size_t index) const;

        std::vector<numa_domain> &numa_domains()
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

        // has to be private bc pointers become invalid after data member
        // thread_pools_ is resized
        // we don't want to allow the user to use it
        detail::init_pool_data const& get_pool_data(
            const std::string &pool_name) const;
        detail::init_pool_data& get_pool_data(std::string const& pool_name);
        detail::init_pool_data& get_default_pool_data();

        void set_scheduler(
            scheduling_policy sched, const std::string &pool_name);

        ////////////////////////////////////////////////////////////////////////
        // counter for instance numbers
        static boost::atomic<int> instance_number_counter_;

        // holds all of the command line switches
        util::command_line_handling cfg_;
        std::size_t cores_needed_;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        std::vector<detail::init_pool_data> initial_thread_pools_;

        // reference to the topology and affinity data
        threads::hwloc_topology_info &topology_;
        hpx::threads::policies::detail::affinity_data affinity_data_;

        // contains the internal topology back-end used to add resources to
        // initial_thread_pools
        std::vector<numa_domain> numa_domains_;
    };

}    // namespace resource
}    // namespace hpx

#endif
