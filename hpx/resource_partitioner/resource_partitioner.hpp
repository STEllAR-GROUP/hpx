//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RESOURCE_PARTITIONER)
#define HPX_RESOURCE_PARTITIONER

#include <hpx/include/runtime.hpp>
#include <hpx/runtime/threads/topology.hpp>
#include <hpx/runtime/threads/detail/thread_pool.hpp>

#include <boost/atomic.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace hpx{

    // structure used to encapsulate all characteristics of thread_pools
    // as specified by the user in int main()
    class initial_thread_pool{
    public:

        initial_thread_pool(std::string name)
                : pool_name_(name)
        {
            if(name.empty())
                throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");
        }

        //! another constructor with size param in case the user already knows at cstrction how many resources will be allocated?

        // get functions
        std::string get_name(){
            return pool_name_;
        }

        std::size_t get_number_pus(){
            return my_pus_.size();
        }

        std::vector<size_t> get_pus(){
            return my_pus_;
        }

        // mechanism for adding resources
        void add_resource(std::size_t pu_number){
            if(pool_name_.empty()){
                std::cout << "[TP - add_resource] empty pool name\n" ;
            }
            std::cout << "adding resource to pool " << pool_name_ << "\n";
            my_pus_.push_back(pu_number);

            //! throw exception if resource does not exist or something ...

        }

    private:
        std::string pool_name_;
        std::vector<std::size_t> my_pus_;
        //! does it need to hold the information "run HPX on me/not"? ie "can be used for runtime"/not?

    };

    class resource_partitioner{
    public:
        resource_partitioner() // queries hwloc, sets internal parameters
//          :
        {
            // allow only one resource_partitioner instance
            if(instance_number_counter_++ >= 0){
                throw std::runtime_error("Cannot instantiate more than one resource partitioner");
            }

            // set pointer to self
            resource_partitioner_ptr = this;

                //! do not allow the creation of more than one RP instance
                //! cf counter for runtime, do the same and add test and exception here

                //! topology is gonna be tricky? see where create_topology is called
                //! probs move it from runtime to RP
        }

        //! constructors with a bunch of strings, in case I know my names already

        // create a new thread_pool, add it to the RP and return a pointer to it
        initial_thread_pool* create_thread_pool(std::string name)
        {
            if(name.empty())
                throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

            initial_thread_pool_.push_back(initial_thread_pool(name));
            initial_thread_pool* ret(&initial_thread_pool_[initial_thread_pool_.size()-1]);
            return ret;
        }

        void add_resource(std::size_t resource, std::string pool_name){
            std::cout << "[RP - add_resource] \n";
            std::cout << "[RP ] adding resource to pool " << get_pool(pool_name)->get_name() << "\n";
            get_pool(pool_name)->add_resource(resource);
        }

        // lots of get_functions
/*        std::size_t get_number_pools(){
            return thread_pools_.size();
        }*/

        static resource_partitioner* get_resource_partitioner_ptr() {
            return resource_partitioner_ptr;
        }

    private:
        ////////////////////////////////////////////////////////////////////////

/*        resource_partitioner* get(){ // for get_ptr function
            return
        }*/


        // has to be private bc pointers become invalid after data member thread_pools_ is resized
        // we don't want to allow the user to use it
        // WARNING will be invalidated
        uint64_t get_pool_index(std::string pool_name){
            std::cout << "[get_pool] \n";
            std::size_t N = initial_thread_pool_.size();
            for(size_t i(0); i<N; i++) {
                if (initial_thread_pool_[i].get_name() == pool_name) {
                    std::cout << "found!\n";
                    return i;
                }
            }

            throw std::invalid_argument(
                    "the resource partitioner does not own a thread pool named \"" + pool_name + "\" \n");

        }

        initial_thread_pool* get_pool(std::string pool_name){
            std::cout << "[get_pool_ptr] \n";

            auto pool = std::find_if(
                    initial_thread_pool_.begin(), initial_thread_pool_.end(),
                    [&pool_name](initial_thread_pool itp) -> bool {return (itp.get_name() == pool_name);}
            );

            if(pool != initial_thread_pool_.end()){
                initial_thread_pool* ret(&(*pool));
                return ret;
            }

            throw std::invalid_argument(
                    "the resource partitioner does not own a thread pool named \"" + pool_name + "\" \n");
        }

        std::size_t get_instance_number_counter() const{
            return static_cast<std::size_t>(instance_number_counter_);
        }

        ////////////////////////////////////////////////////////////////////////

        // counter for instance numbers
        static boost::atomic<int> instance_number_counter_;

        // pointer to global unique instance of resource_partitioner
        static resource_partitioner* resource_partitioner_ptr;

        // contains the basic characteristics of the thread pool partitioning ...
        // that will be passed to the runtime
        //! instead of a struct, should I just have a map of names to vector<size_t>??
        std::vector<initial_thread_pool> initial_thread_pool_;

        // actual thread pools of OS-threads
//        std::vector<threads::detail::thread_pool> thread_pools_; //! template param?

        // list of schedulers or is it enough if they're owned by thread_pool?

        // reference to the topology
//        threads::topology& topology_;

        // reference to affinity data


    };

/*    resource_partitioner * get_resource_partitioner_ptr() const
    {
        //! throw exception if this is called before the RP has been instanciated
        // if resource_partitioner has been instanciated already (most cases)
        resource_partitioner** rp = runtime::runtime_.get();
        return rp ? *rp : nullptr;
    }*/

    boost::atomic<int> resource_partitioner::instance_number_counter_(-1); //! move to .cpp

    resource_partitioner* resource_partitioner::resource_partitioner_ptr(nullptr); //! move to .cpp probably


}



#endif