//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER)
#define HPX_RESOURCE_PARTITIONER

#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/runtime/threads/policies/hwloc_topology_info.hpp>
#include <hpx/runtime/threads/policies/affinity_data.hpp>
#include <hpx/util/thread_specific_ptr.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>

//#include <hpx/runtime/threads/detail/thread_pool.hpp>
//#include <hpx/include/runtime.hpp>

#include <boost/atomic.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace hpx {


    namespace resource {

    // scheduler assigned to thread_pool
    // choose same names as in command-line options except with _ instead of -
    enum scheduling_policy {
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

        init_pool_data(std::string name, scheduling_policy = scheduling_policy::unspecified);

        //! another constructor with size param in case the user already knows at cstrction how many resources will be allocated?
        //! this constructor would call "reserve" on data member and be a little more mem-efficient

        // set functions
        void set_scheduler(scheduling_policy sched);

        // get functions
        std::string get_name() const;
        scheduling_policy get_scheduling_policy() const;
        std::size_t get_number_pus() const;
        std::vector<size_t> get_pus() const;

        // mechanism for adding resources
        void add_resource(std::size_t pu_number);

    private:
        std::string pool_name_;
        scheduling_policy scheduling_policy_;
        std::vector<std::size_t> my_pus_;
        //! does it need to hold the information "run HPX on me/not"? ie "can be used for runtime"/not?
        //! would make life easier for ppl who want to run HPX side-by-sie with OpenMP for example?

    };

    class HPX_EXPORT resource_partitioner{
    public:

/*        static resource_partitioner& get_resource_partitioner_(){
            util::static_<resource::resource_partitioner, std::false_type> rp;
            return rp.get();
        }*/

        //! constructor
        //! users shouldn't use the constructor but rather get_resource_partitioner
        resource_partitioner();


        // create a new thread_pool, add it to the RP and return a pointer to it
        void create_thread_pool(std::string name, scheduling_policy sched = scheduling_policy::unspecified);
        void create_default_pool(scheduling_policy sched = scheduling_policy::unspecified);

        //! called in hpx_init run_or_start
        void set_init_affinity_data(hpx::threads::policies::init_affinity_data init_affdat);
        void set_default_pool(std::size_t num_threads);
        void set_default_schedulers(std::string queueing);

        //! setup stuff related to pools
        void add_resource(std::size_t resource, std::string pool_name);
        void add_resource_to_default(std::size_t resource);
        void set_scheduler(scheduling_policy sched, std::string pool_name);

        // called in hpx_init
        void init();

        // lots of get_functions
/*        std::size_t get_number_pools(){
            return thread_pools_.size();
            // get pointer to a particular pool?
        }*/

        threads::topology& get_topology() const;
        threads::policies::init_affinity_data get_init_affinity_data() const;


    private:

        ////////////////////////////////////////////////////////////////////////

        //! this is ugly, I should probably delete it
        uint64_t get_pool_index(std::string pool_name);

        // has to be private bc pointers become invalid after data member thread_pools_ is resized
        // we don't want to allow the user to use it
        init_pool_data* get_pool(std::string pool_name);

        ////////////////////////////////////////////////////////////////////////

        // counter for instance numbers
        static boost::atomic<int> instance_number_counter_;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        std::vector<init_pool_data> initial_thread_pools_;

        // actual thread pools of OS-threads
//        std::vector<threads::detail::thread_pool> thread_pools_; //! template param needed? owned via thread_manager? Different data structure?

        // reference to the topology
        threads::hwloc_topology_info& topology_;

        // reference to affinity data
        //! I'll probably have to take this away from runtime
        hpx::threads::policies::init_affinity_data init_affinity_data_;
        hpx::threads::policies::detail::affinity_data affinity_data_;


    };

    } // namespace resource

    static resource::resource_partitioner & get_resource_partitioner()
    {
        util::static_<resource::resource_partitioner, std::false_type> rp;
        return rp.get();
        //return resource::resource_partitioner::get_resource_partitioner_();
    }

} // namespace hpx



#endif
