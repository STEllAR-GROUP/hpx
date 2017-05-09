//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>

namespace hpx { namespace resource {

    ////////////////////////////////////////////////////////////////////////

    initial_thread_pool::initial_thread_pool(std::string name)
        : pool_name_(name)
    {
        if(name.empty())
        throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");
    };

    std::string initial_thread_pool::get_name(){
        return pool_name_;
    }

    std::size_t initial_thread_pool::get_number_pus(){
        return my_pus_.size();
    }

    std::vector<size_t> initial_thread_pool::get_pus(){
        return my_pus_;
    }

    // mechanism for adding resources
    void initial_thread_pool::add_resource(std::size_t pu_number){
        my_pus_.push_back(pu_number);

        //! throw exception if resource does not exist
        //! or the input parameter is invalid or something like that ...

    }

    ////////////////////////////////////////////////////////////////////////

    resource_partitioner::resource_partitioner()
        : topology_(threads::create_topology())
    {
        // initialize our TSS
        resource_partitioner::init_tss();

        // allow only one resource_partitioner instance
        if(instance_number_counter_++ >= 0){
            throw std::runtime_error("Cannot instantiate more than one resource partitioner");
        }

        // set pointer to self
        resource_partitioner_ptr = this;
    }

    // create a new thread_pool, add it to the RP and return a pointer to it
    initial_thread_pool* resource_partitioner::create_thread_pool(std::string name)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

        initial_thread_pool_.push_back(initial_thread_pool(name));
        initial_thread_pool* ret(&initial_thread_pool_[initial_thread_pool_.size()-1]);
        return ret;
    }

    void resource_partitioner::add_resource(std::size_t resource, std::string pool_name){
        get_pool(pool_name)->add_resource(resource);
    }

    threads::topology& resource_partitioner::get_topology() const
    {
        return topology_;
    }

    ////////////////////////////////////////////////////////////////////////

    util::thread_specific_ptr<resource_partitioner*, resource_partitioner::tls_tag> resource_partitioner::resource_partitioner_;

    void resource_partitioner::init_tss()
    {
        // initialize our TSS
        if (nullptr == resource_partitioner::resource_partitioner_.get())
        {
            HPX_ASSERT(nullptr == threads::thread_self::get_self());
            resource_partitioner::resource_partitioner_.reset(new resource_partitioner* (this));
            //!threads::thread_self::init_self();//!
        }
    }

    //! never called ... ?!?
/*    void resource_partitioner::deinit_tss()
    {
        // reset our TSS
        threads::thread_self::reset_self();
        util::reset_held_lock_data();
        threads::reset_continuation_recursion_count();
    }*/



    // if resource manager has not been instantiated yet, it simply returns a nullptr
    resource_partitioner* get_resource_partitioner_ptr() {
        resource_partitioner** rp = resource_partitioner::resource_partitioner_.get();
        return rp ? *rp : nullptr;
    }
/*    resource_partitioner* resource_partitioner::get_resource_partitioner_ptr() {
        return resource_partitioner_ptr;
    }*/

    ////////////////////////////////////////////////////////////////////////

    uint64_t resource_partitioner::get_pool_index(std::string pool_name){
        std::size_t N = initial_thread_pool_.size();
        for(size_t i(0); i<N; i++) {
            if (initial_thread_pool_[i].get_name() == pool_name) {
                return i;
            }
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a thread pool named \"" + pool_name + "\" \n");

    }

    // has to be private bc pointers become invalid after data member thread_pools_ is resized
    // we don't want to allow the user to use it
    initial_thread_pool* resource_partitioner::get_pool(std::string pool_name){
        auto pool = std::find_if(
                initial_thread_pool_.begin(), initial_thread_pool_.end(),
                [&pool_name](initial_thread_pool itp) -> bool {return (itp.get_name() == pool_name);}
        );

        if(pool != initial_thread_pool_.end()){
            initial_thread_pool* ret(&(*pool)); //! ugly
            return ret;
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a thread pool named \"" + pool_name + "\" \n");
    }

    ////////////////////////////////////////////////////////////////////////

    boost::atomic<int> resource_partitioner::instance_number_counter_(-1); //! move to .cpp

    resource_partitioner* resource_partitioner::resource_partitioner_ptr(nullptr); //! move to .cpp probably


} }