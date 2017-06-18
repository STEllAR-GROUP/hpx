//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER)
#define HPX_RESOURCE_PARTITIONER

#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology_info.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
//
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/policies/callback_notifier.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>


namespace hpx {

    namespace resource {
        class resource_partitioner;
    }

    // if the resource partitioner is accessed before the HPX runtime has started
    // then on first access, this function should be used, to ensure that command line
    // affinity binding options are honoured. Use this function signature only once
    // and thereafter use the parameter free version.
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            int argc, char **argv);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, bool check=true);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            int argc, char **argv, std::vector<std::string> ini_config);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            int argc, char **argv, runtime_mode mode);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, std::vector<std::string> ini_config);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, runtime_mode mode);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            int argc, char **argv, std::vector<std::string> ini_config,
            runtime_mode mode);
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner(
            boost::program_options::options_description desc_cmdline,
            int argc, char **argv, std::vector<std::string> ini_config,
            runtime_mode mode, bool check = true);

    // may be used anywhere in code and returns a reference to the
    // single, global resource partitioner
    extern HPX_EXPORT hpx::resource::resource_partitioner & get_resource_partitioner();

namespace resource
{

    struct core;
    struct numa_domain;

    struct pu {

    private:
        std::size_t     id_;
        core           *core_;
        std::vector<pu> pus_sharing_core();
        std::vector<pu> pus_sharing_numa_domain();
        std::size_t     thread_occupancy_;
                        // indicates the number of threads that should run on this PU
                        //  0: this PU is not exposed by the affinity bindings
                        //  1: normal occupancy
                        // >1: oversubscription
        mutable std::size_t     thread_occupancy_count_;
                        // counts number of threads bound to this PU
        friend struct core;
        friend struct numa_domain;
        friend class resource_partitioner;

    public:
        std::size_t     id() const {
            return id_;
        }
    };

    struct core {
    private:
        std::size_t       id_;
        numa_domain      *domain_;
        std::vector<pu>   pus_;
        std::vector<core> cores_sharing_numa_domain();
        friend struct pu;
        friend struct numa_domain;
        friend class resource_partitioner;
    public:
        const std::vector<pu> &pus() const {
            return pus_;
        }
        std::size_t     id() const {
            return id_;
        }
    };

    struct numa_domain {
    private:
        std::size_t       id_;
        std::vector<core> cores_;
        friend struct pu;
        friend struct core;
        friend class resource_partitioner;
    public:
        const std::vector<core> &cores() const {
            return cores_;
        }
        std::size_t     id() const {
            return id_;
        }
    };

    using scheduler_function = std::function<
        hpx::threads::detail::thread_pool*(
            hpx::threads::policies::callback_notifier &notifier,
            std::size_t index, char const* name,
            hpx::threads::policies::scheduler_mode m,
            std::size_t thread_offset)>;

    // scheduler assigned to thread_pool
    // choose same names as in command-line options except with _ instead of -
    enum scheduling_policy {
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

    // structure used to encapsulate all characteristics of thread_pools
    // as specified by the user in int main()
    class init_pool_data {
    public:

        // mechanism for adding resources (zero-based index)
        void add_resource(std::size_t pu_index, std::size_t num_threads);

        void print_pool() const;

        friend class resource_partitioner;

        static std::size_t  num_threads_overall;        // counter ... overall, in all the thread pools

    private:

        init_pool_data(const std::string &name,
                       scheduling_policy = scheduling_policy::unspecified);

        init_pool_data(const std::string &name,
                       scheduler_function create_func);

        std::string         pool_name_;
        scheduling_policy   scheduling_policy_;
        std::vector<threads::mask_type>  assigned_pus_; // PUs this pool is allowed to run on
        std::size_t         num_threads_;               // counter for number of threads bound to this pool
        scheduler_function  create_function_;
    };

    class HPX_EXPORT resource_partitioner{
    public:

        // constructor: users shouldn't use the constructor
        // but rather get_resource_partitioner
        resource_partitioner();

        void set_hpx_init_options(
                util::function_nonser<
                        int(boost::program_options::variables_map& vm)
                > const& f);

        int call_cmd_line_options(
                boost::program_options::options_description const& desc_cmdline,
                int argc, char** argv);

        bool pu_exposed(std::size_t pid);

        void print_init_pool_data() const;

        // create a thread_pool
        void create_thread_pool(const std::string &name,
            scheduling_policy sched = scheduling_policy::unspecified);

        // create a thread_pool with a callback function for creating a custom scheduler
        void create_thread_pool(const std::string &name,
            scheduler_function scheduler_creation);

        // Functions to add processing units to thread pools via
        // the pu/core/numa_domain API
        void add_resource(const hpx::resource::pu &p,
            const std::string &pool_name, std::size_t num_threads = 1);
        void add_resource(const std::vector<hpx::resource::pu> &pv,
            const std::string &pool_name);
        void add_resource(const hpx::resource::core &c,
            const std::string &pool_name);
        void add_resource(const std::vector<hpx::resource::core> &cv,
            const std::string &pool_name);
        void add_resource(const hpx::resource::numa_domain &nd,
            const std::string &pool_name);
        void add_resource(const std::vector<hpx::resource::numa_domain> &ndv,
            const std::string &pool_name);

        // stuff that has to be done during hpx_init ...
        void set_scheduler(scheduling_policy sched, const std::string &pool_name);
        void set_threadmanager(threads::threadmanager_base* thrd_manag);
        threads::threadmanager_base* get_thread_manager() const;

        // called by constructor of scheduler_base
        threads::policies::detail::affinity_data* get_affinity_data() {
            return &affinity_data_;
        }

        // Does initialization of all resources and internal data of the resource partitioner
        // called in hpx_init
        void configure_pools();

        // called in runtime::assign_cores()
        size_t init() {
            thread_manager_->init();
            return cores_needed_;
        }

        ////////////////////////////////////////////////////////////////////////

        scheduling_policy which_scheduler(const std::string &pool_name);
        threads::topology& get_topology() const;
        util::command_line_handling& get_command_line_switches();
        std::size_t get_num_threads() const;
        size_t get_num_pools() const;
        size_t get_num_threads(const std::string &pool_name);
        init_pool_data* get_pool(std::size_t pool_index);
        const std::string &get_pool_name(size_t index) const;
        init_pool_data* get_pool(std::size_t pool_index) const;
        threads::mask_cref_type get_pu_mask(std::size_t num_thread, bool numa_sensitive) const;
        bool cmd_line_parsed() const;
        int parse(
                boost::program_options::options_description desc_cmdline,
                int argc, char **argv, std::vector<std::string> ini_config,
                runtime_mode mode, bool fill_internal_topology = true);

        const scheduler_function &get_pool_creator(size_t index) const;

        std::vector<numa_domain> &numa_domains() {
            return numa_domains_;
        }

        bool terminate_after_parse() {
            return cfg_.parse_terminate_;
        }

    private:

        ////////////////////////////////////////////////////////////////////////

        void fill_topology_vectors();

        ////////////////////////////////////////////////////////////////////////

        // called in hpx_init run_or_start
        void set_init_affinity_data();
        void setup_pools();
        void setup_schedulers();
        void reconfigure_affinities();
        bool check_empty_pools() const;


        //! this is ugly, I should probably delete it
        uint64_t get_pool_index(const std::string &pool_name) const;

        // has to be private bc pointers become invalid after data member thread_pools_ is resized
        // we don't want to allow the user to use it
        init_pool_data* get_pool(const std::string &pool_name);
        init_pool_data* get_default_pool();

        ////////////////////////////////////////////////////////////////////////

        // counter for instance numbers
        static boost::atomic<int> instance_number_counter_;

        // holds all of the command line switches
        util::command_line_handling cfg_;
        std::size_t cores_needed_;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        std::vector<init_pool_data> initial_thread_pools_;

        // pointer to the threadmanager instance
        hpx::threads::threadmanager_base* thread_manager_;

        // reference to the topology and affinity data
        threads::hwloc_topology_info& topology_;
        hpx::threads::policies::detail::affinity_data affinity_data_;

        // contains the internal topology backend used to add resources to initial_thread_pools
        std::vector<numa_domain> numa_domains_;

    };

    } // namespace resource
} // namespace hpx



#endif
