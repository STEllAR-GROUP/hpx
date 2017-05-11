//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/include/runtime.hpp>

namespace hpx { namespace resource {

    ////////////////////////////////////////////////////////////////////////

    init_pool_data::init_pool_data(std::string name, scheduling_policy sched)
        : pool_name_(name),
          scheduling_policy_(sched)
    {
        if(name.empty())
        throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");
    };

    std::string init_pool_data::get_name() const {
        return pool_name_;
    }

    scheduling_policy init_pool_data::get_scheduling_policy() const {
        return scheduling_policy_;
    }

    std::size_t init_pool_data::get_number_pus() const {
        return my_pus_.size();
    }

    std::vector<size_t> init_pool_data::get_pus() const {
        return my_pus_;
    }

    // mechanism for adding resources
    void init_pool_data::add_resource(std::size_t pu_number){
        my_pus_.push_back(pu_number);

        //! throw exception if resource does not exist
        //! or the input parameter is invalid or something like that ...

    }

    void init_pool_data::set_scheduler(scheduling_policy sched){
        scheduling_policy_ = sched;
    }

    void print_vector(std::vector<size_t> const& v){
        std::size_t s = v.size();
        if(s==0) {
            std::cout << "(empty)\n";
            return;
        }
        std::cout << v[0] ;
        for(size_t i(1); i<s ;i++){
            std::cout << ", " << v[i];
        }
        std::cout << "\n";
    }

    void init_pool_data::print_me(){
        std::cout << "[pool \"" << pool_name_ << "\"] with scheduler " ;
        std::string sched;
        switch(scheduling_policy_) {
            case -1 :
                sched = "unspecified"; break;
            case 0 :
                sched = "local"; break;
            case 1 :
                sched = "local_priority_fifo"; break;
            case 2 :
                sched = "local_priority_lifo"; break;
            case 3 :
                sched = "static"; break;
            case 4 :
                sched = "static_priority"; break;
            case 5 :
                sched = "abp_priority"; break;
            case 6 :
                sched = "hierarchy"; break;
            case 7 :
                sched = "periodic_priority"; break;
            case 8 :
                sched = "throttle"; break;
        }
        std::cout << "\"" << sched << "\"\n"
                  << "is running on PUs : ";
        print_vector(my_pus_);
    }


    ////////////////////////////////////////////////////////////////////////

    resource_partitioner::resource_partitioner()
            : topology_(threads::create_topology())
    {
        // allow only one resource_partitioner instance
        if(instance_number_counter_++ >= 0){
            throw std::runtime_error("Cannot instantiate more than one resource partitioner");
        }
    }

    void resource_partitioner::set_init_affinity_data(hpx::threads::policies::init_affinity_data init_affdat){
        init_affinity_data_ = init_affdat;
    }

    void resource_partitioner::set_default_pool(std::size_t num_threads) {
        //! check whether the user created a default_pool already. If so, do nothing.
        if(default_pool())
            return;

        //! if the user did not create one yet, do so now
        create_default_pool();


        //! If the user specified a default_pool, but there still are some unassigned resources,
        //! should they be added to the default pool or simply ignored (current implementation: ignored)

        //! take all non-assigned resources and throw them in a regular default pool

        //! FIXME Ugh, how exactly should this interact with num_threads_ specified in cmd line??

        std::size_t num_pus = topology_.get_number_of_pus();
        std::vector<size_t> pu_usage(num_pus);
        for(auto itp : initial_thread_pools_){
            for(auto pu : itp.get_pus()){
                pu_usage[pu] ++;
            }
        }

        std::cout << "PU usage: "; print_vector(pu_usage); //! to delete

        for(size_t i(0); i<num_pus; i++){
            if(pu_usage[i] == 0)
                add_resource_to_default(i);
        }

    }

    void resource_partitioner::set_default_schedulers(std::string queueing) {

        // select the default scheduler
        scheduling_policy default_scheduler;

        if (0 == std::string("local").find(queueing))
        {
            default_scheduler = scheduling_policy::local;
        }
        else if (0 == std::string("local-priority-fifo").find(queueing))
        {
            default_scheduler = scheduling_policy::local_priority_fifo ;
        }
        else if (0 == std::string("local-priority-lifo").find(queueing))
        {
            default_scheduler = scheduling_policy::local_priority_lifo;
        }
        else if (0 == std::string("static").find(queueing))
        {
            default_scheduler = scheduling_policy::static_;
        }
        else if (0 == std::string("static-priority").find(queueing))
        {
            default_scheduler = scheduling_policy::static_priority;
        }
        else if (0 == std::string("abp-priority").find(queueing))
        {
            default_scheduler = scheduling_policy::abp_priority;
        }
        else if (0 == std::string("hierarchy").find(queueing))
        {
            default_scheduler = scheduling_policy::hierarchy;
        }
        else if (0 == std::string("periodic-priority").find(queueing))
        {
            default_scheduler = scheduling_policy::periodic_priority;
        }
        else if (0 == std::string("throttle").find(queueing)) {
            default_scheduler = scheduling_policy::throttle;
        }
        else {
            throw detail::command_line_error(
                    "Bad value for command line option --hpx:queuing");
        }

        // set this scheduler on the pools that do not have a specified scheduler yet
        std::size_t npools(initial_thread_pools_.size());
        for(size_t i(0); i<npools; i++){
            if(initial_thread_pools_[i].get_scheduling_policy() == unspecified){
                initial_thread_pools_[i].set_scheduler(default_scheduler);
            }
        }
    }


    // create a new thread_pool, add it to the RP and return a pointer to it
    void resource_partitioner::create_thread_pool(std::string name, scheduling_policy sched)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

        initial_thread_pools_.push_back(init_pool_data(name, sched));
        /*init_pool_data* ret(&initial_thread_pools_[initial_thread_pools_.size()-1]);
        return ret;*/ //! or should I return a pointer to that pool?
    }
    void resource_partitioner::create_default_pool(scheduling_policy sched) {
        create_thread_pool("default", sched);
    }


    void resource_partitioner::add_resource(std::size_t resource, std::string pool_name){
        get_pool(pool_name)->add_resource(resource);
    }
    void resource_partitioner::add_resource_to_default(std::size_t resource){
        add_resource(resource, "default");
    }

    void resource_partitioner::set_scheduler(scheduling_policy sched, std::string pool_name){
        get_pool(pool_name)->set_scheduler(sched);
    }

    void resource_partitioner::init(){

        //! what do I actually need to do?

        //! en gros:

        //! if nothing else, than just take all previous little setups and stick them here

    }


    threads::topology& resource_partitioner::get_topology() const
    {
        return topology_;
    }

    threads::policies::init_affinity_data resource_partitioner::get_init_affinity_data() const
    { //! should this return a pointer instead of a copy?
        return init_affinity_data_;
    }


    ////////////////////////////////////////////////////////////////////////

    uint64_t resource_partitioner::get_pool_index(std::string pool_name){
        std::size_t N = initial_thread_pools_.size();
        for(size_t i(0); i<N; i++) {
            if (initial_thread_pools_[i].get_name() == pool_name) {
                return i;
            }
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a thread pool named \"" + pool_name + "\" \n");

    }

    // has to be private bc pointers become invalid after data member thread_pools_ is resized
    // we don't want to allow the user to use it
    init_pool_data* resource_partitioner::get_pool(std::string pool_name){
        auto pool = std::find_if(
                initial_thread_pools_.begin(), initial_thread_pools_.end(),
                [&pool_name](init_pool_data itp) -> bool {return (itp.get_name() == pool_name);}
        );

        if(pool != initial_thread_pools_.end()){
            init_pool_data* ret(&(*pool)); //! FIXME
            return ret;
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a thread pool named \"" + pool_name + "\". \n");
        //! Add names of available pools?
    }

    init_pool_data* resource_partitioner::get_default_pool(){
        auto pool = std::find_if(
                initial_thread_pools_.begin(), initial_thread_pools_.end(),
                [](init_pool_data itp) -> bool {return (itp.get_name() == "default");}
        );

        if(pool != initial_thread_pools_.end()){
            init_pool_data* ret(&(*pool)); //! FIXME
            return ret;
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a default pool \n");
    }

    bool resource_partitioner::default_pool(){
        auto pool = std::find_if(
                initial_thread_pools_.begin(), initial_thread_pools_.end(),
                [](init_pool_data itp) -> bool {return (itp.get_name() == "default");}
        );

        if(pool != initial_thread_pools_.end()){
            return true;
        }
        return false;
    }


    void resource_partitioner::print_me(){
        std::cout << "the resource partitioner owns "
                  << initial_thread_pools_.size() << " pool(s) : \n";
        for(auto itp : initial_thread_pools_){
            itp.print_me();
        }
    }

    ////////////////////////////////////////////////////////////////////////

    boost::atomic<int> resource_partitioner::instance_number_counter_(-1);

    } // namespace resource

} // namespace hpx
