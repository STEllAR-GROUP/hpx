//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DETAIL_RESOURCE_PARTITIONER_AUG_10_2017_0926AM)
#define HPX_DETAIL_RESOURCE_PARTITIONER_AUG_10_2017_0926AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/resource/partitioner.hpp>
#include <hpx/runtime/runtime_mode.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/util/command_line_handling.hpp>
#include <hpx/util/tuple.hpp>

#include <boost/atomic.hpp>
#include <boost/program_options.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace hpx { namespace resource { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
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

        friend class resource::detail::partitioner;

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

    ///////////////////////////////////////////////////////////////////////
    class partitioner
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        partitioner();

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
            resource::partitioner_mode rpmode,
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
        threads::topology &topology_;
        hpx::threads::policies::detail::affinity_data affinity_data_;

        // contains the internal topology back-end used to add resources to
        // initial_thread_pools
        std::vector<numa_domain> numa_domains_;

        // store policy flags determining the general behavior of the
        // resource_partitioner
        resource::partitioner_mode mode_;
    };
}}}

#endif
