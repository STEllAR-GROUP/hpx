//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/include/runtime.hpp>
//
#include <bitset>

#if defined(HPX_HAVE_MAX_CPU_COUNT)
    typedef std::bitset<HPX_HAVE_MAX_CPU_COUNT> bitset_type;
#else
    typedef std::bitset<32> bitset_type;
#endif

namespace hpx {

    namespace detail {

        std::size_t get_pu_offset(util::command_line_handling const& cfg)
        {
            std::size_t pu_offset = std::size_t(-1);
#if defined(HPX_HAVE_HWLOC)
            if (cfg.pu_offset_ != std::size_t(-1))
            {
                pu_offset = cfg.pu_offset_;
                if (pu_offset >= hpx::threads::hardware_concurrency())
                {
                    throw detail::command_line_error(
                            "Invalid command line option "
                                    "--hpx:pu-offset, value must be smaller than number of "
                                    "available processing units.");
                }
            }
#endif
            return pu_offset;
        }

        std::size_t get_pu_step(util::command_line_handling const& cfg)
        {
            std::size_t pu_step = 1;
#if defined(HPX_HAVE_HWLOC)
            if (cfg.pu_step_ != 1) {
                pu_step = cfg.pu_step_;
                if (pu_step == 0 || pu_step >= hpx::threads::hardware_concurrency())
                {
                    throw detail::command_line_error(
                            "Invalid command line option "
                                    "--hpx:pu-step, value must be non-zero and smaller than "
                                    "number of available processing units.");
                }
            }
#endif
            return pu_step;
        }

    } // namespace detail

namespace resource
{

    std::vector<pu> pu::pus_sharing_core()
    {
        std::vector<pu> result;
        for (const pu &p : core_->pus_) {
            if (p.id_!=id_) {
                result.push_back(p);
            }
        }
        return result;
    }

    std::vector<pu> pu::pus_sharing_numa_domain()
    {
        std::vector<pu> result;
        for (const core &c : core_->domain_->cores_) {
            for (const pu &p : c.pus_) {
                if (p.id_!=id_) {
                    result.push_back(p);
                }
            }
        }
        return result;
    }

    std::vector<core> core::cores_sharing_numa_domain()
    {
        std::vector<core> result;
        for (const core &c : domain_->cores_) {
            if (c.id_!=id_) {
                result.push_back(c);
            }
        }
        return result;
    }

    ////////////////////////////////////////////////////////////////////////

    init_pool_data::init_pool_data(const std::string &name, scheduling_policy sched)
        : pool_name_(name),
          scheduling_policy_(sched),
          assigned_pus_(0),
          num_threads_(0)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");
        threads::resize(assigned_pus_, hpx::threads::hardware_concurrency());
    };

    const std::string &init_pool_data::get_name() const {
        return pool_name_;
    }

    scheduling_policy init_pool_data::get_scheduling_policy() const {
        return scheduling_policy_;
    }

    std::size_t init_pool_data::get_number_used_pus() const {
        return threads::count(assigned_pus_);
    }

    std::size_t init_pool_data::get_num_threads() const {
        return num_threads_;
    }

    threads::mask_type init_pool_data::get_pus() const {
        return assigned_pus_;
    }

    // mechanism for adding resources
    void init_pool_data::add_resource(std::size_t pu_index){
        if (pu_index >= hpx::threads::hardware_concurrency()) {
            throw std::invalid_argument("Processing unit index out of bounds."); //! FIXME give actual number of PUs
        }

        threads::set(assigned_pus_, pu_index);
    }

    void init_pool_data::set_scheduler(scheduling_policy sched){
        scheduling_policy_ = sched;
    }

    void init_pool_data::set_mask(threads::mask_type mask){
        assigned_pus_ = mask;
    }

    void init_pool_data::set_thread_num(std::size_t num_threads){
        num_threads_ = num_threads;
    }

    void init_pool_data::print_pool() const {
        std::cout << "[pool \"" << pool_name_ << "\"] with scheduler " ;
        std::string sched;
        switch(scheduling_policy_) {
            case -1 : sched = "unspecified";        break;
            case 0 : sched = "local";               break;
            case 1 : sched = "local_priority_fifo"; break;
            case 2 : sched = "local_priority_lifo"; break;
            case 3 : sched = "static";              break;
            case 4 : sched = "static_priority";     break;
            case 5 : sched = "abp_priority";        break;
            case 6 : sched = "hierarchy";           break;
            case 7 : sched = "periodic_priority";   break;
            case 8 : sched = "throttle";            break;
        }
        std::cout << "\"" << sched << "\"\n"
                  << "is running on PUs : ";
        std::cout << bitset_type(assigned_pus_) << "\n";
    }


    ////////////////////////////////////////////////////////////////////////

    resource_partitioner::resource_partitioner(std::size_t num_special_pools_)
            : thread_manager_(nullptr),
              topology_(threads::create_topology()),
              set_affinity_from_resource_partitioner_(false)
    {
        // Reserve the appropriate size of initial thread pools
        initial_thread_pools_.reserve(num_special_pools_ + 1);

        // Create the default pool
        initial_thread_pools_.push_back(init_pool_data("default"));

        // allow only one resource_partitioner instance
        if(instance_number_counter_++ >= 0)
            throw std::runtime_error("Cannot instantiate more than one resource partitioner");

        fill_topology_vectors();
    }

    void resource_partitioner::set_init_affinity_data(hpx::util::command_line_handling const& cfg)
    {
        // Setup the initial affinity data
        std::size_t pu_offset = hpx::detail::get_pu_offset(cfg);
        std::size_t pu_step = hpx::detail::get_pu_step(cfg);
        std::string affinity_domain = hpx::detail::get_affinity_domain(cfg);

        // if the binding should be set from the user's instructions in int main()
        std::string affinity_desc;
        if(set_affinity_from_resource_partitioner_){
            affinity_desc = "affinity-from-resource-partitioner";
        } else {
            std::size_t numa_sensitive = hpx::detail::get_affinity_description(cfg, affinity_desc);
        }

        init_affinity_data_ = threads::policies::init_affinity_data(
                pu_offset, pu_step, affinity_domain, affinity_desc);
    }

    void resource_partitioner::fill_topology_vectors()
    {
        std::size_t pid = 0;
        std::size_t N = topology_.get_number_of_numa_nodes();
        if (N==0) N= topology_.get_number_of_sockets();
        numa_domains_.reserve(N);
        //
        for (std::size_t i=0; i<N; ++i) {
            numa_domains_.push_back(numa_domain());
            numa_domain &nd = numa_domains_.back();
            nd.id_          = i;
            nd.cores_.reserve(topology_.get_number_of_numa_node_cores(i));
            for (std::size_t j=0; j<topology_.get_number_of_numa_node_cores(i); ++j) {
                nd.cores_.push_back(core());
                core &c   = nd.cores_.back();
                c.id_     = j;
                c.domain_ = &nd;
                c.pus_.reserve(topology_.get_number_of_core_pus(j));
                for (std::size_t k=0; k<topology_.get_number_of_core_pus(j); ++k) {
                    c.pus_.push_back(pu());
                    pu &p   = c.pus_.back();
                    p.id_   = pid;
                    p.core_ = &c;
                    pid++;
                    std::cout << "domain " << nd.id_ << " core " << c.id_ << " pu_id " << p.id_ << "\n";
                }
            }
        }
    }

    // this function is called in hpx_init, even before the instantiation of the runtime
    // attributes to default pool all resources that have not been
    // attributed to any other pool.
    // well, in the meantime, this functions does much more:
    // -2 checks whether there are oversubscribed PUs
    // -3 sets data member "desired_number_threads" for each pool
    //! FIXME rename this function
    void resource_partitioner::set_default_pool(std::size_t num_threads) {

        // check whether any of the pools defined up to now are empty
        // note: does not check "default", this one is allowed not to be given resources by the user
        if(check_empty_pools())
            throw std::invalid_argument("Pools empty of resources are not allowed. Please re-run this program with allow-empty-pool-policy (not implemented yet)");
        //! FIXME add allow-empty-pools policy. Wait, does this even make sense??

        // check whether any of the PUs are oversubscribed
        if(check_oversubscription())
            throw std::invalid_argument("Oversubscription of hardware processing units is not allowed. If you want to oversubscribe, please use the policy allow-oversubscription (not implemented yet)");
        //! FIXME add allow-oversubscription policy

        // assign the desired number of threads to each pool
        // and calculate the total number of OS-threads to be instantiated
        // by thread_pool_impl in run()
        std::size_t num_threads_desired_total = 0;
        std::size_t thread_num = 0;
        for(auto& itp : initial_thread_pools_){
            thread_num = threads::count(itp.get_pus());
            num_threads_desired_total += thread_num;
            itp.set_thread_num(thread_num);
        }

        // make sure the sum of the number of desired threads is strictly smaller
        // than the total number of OS-threads that will be created (specified by --hpx:threads)
        if(num_threads_desired_total > affinity_data_.get_num_threads()){
            throw std::invalid_argument("The desired number of threads is greater than the number of threads provided in the command line. \n");
            //! FIXME give indication: --hpx:threads N >= (num_threads_desired_total+1)
        }

        // If the default pool already has resources assigned to it (by the user),
        // set its number of threads (all that's left over from the number specified in
        // command line.
        //! FIXME is this right, or should we add all free resources to the default pool,
        //! FIXME even though the user has not assigned these to the default pool himself?
        if(threads::any(get_default_pool()->get_pus())) {
            return;
        }

        // Get a mask of all used PUs by doing a bitwise or on all the masks
        std::size_t num_pus = topology_.get_number_of_pus();
        threads::mask_type cummulated_pu_usage = threads::mask_type(0);
        for(auto itp : initial_thread_pools_) {
            cummulated_pu_usage = cummulated_pu_usage | itp.get_pus();
        }

        // If the user did not assign any resources to the default pool
        //! FIXME should I do this even if the user did assign some resources, but there still are some left?
        // take all resources that have not been assigned to any thread pool yet
        // and give them to the default pool
        threads::mask_type default_mask = threads::not_(cummulated_pu_usage);
        //! cut off the bits that are above hardware concurrency ...
        default_mask &= threads::mask_type((1<<topology_.get_number_of_pus()) - 1);

        // make sure the mask for the default pool has resources in it
        //! FIXME add allow-empty-default-pool policy
        if(!threads::any(default_mask))
            throw std::invalid_argument("No processing units left over for the default pool. If you want to allow an empty pool, use allow-empty-default-pool policy");

        // set default mask
        get_default_pool()->set_mask(default_mask);

        // compute number of threads for default and set it
        std::size_t num_available_threads_for_default = threads::count(default_mask); //! #PUs that the default pool can use
        std::size_t num_threads_default_pool = affinity_data_.get_num_threads() - num_threads_desired_total; //! hpx:threads - (threads used for other pools)

        if(num_threads_default_pool > num_available_threads_for_default){
            //! throw oversubscription exception except if policy blahblah
        } else if (num_threads_default_pool == num_available_threads_for_default){
            //! it's all fine :) just set that number :)
        } else { //! num_threads_default_pool < num_available_threads_for_default
            //! reduce the mask to the first available resources #bits
            std::size_t excess_pus = num_available_threads_for_default - num_threads_default_pool;
            while(excess_pus > 0){
                threads::unset(default_mask, threads::find_first(default_mask));
                excess_pus--;
            }
            // re-set default mask since it has been modified
            get_default_pool()->set_mask(default_mask);
            //! FIXME reduce in a smart way: eg try to take PUs that are close together or something like that
            //! current implementation = just unset the number of bits required in order of appearance in the mask...
        }

        get_default_pool()->set_thread_num(num_threads_default_pool);

    }

    void resource_partitioner::set_default_schedulers(const std::string &queueing) {

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

    // Returns true if any processing unit has been assigned to two different thread-pools
    // called in set_default_pool()
    bool resource_partitioner::check_oversubscription() const
    {
        threads::mask_type pus_in_common((1<<(topology_.get_number_of_pus()))-1);

        for(auto& itp : initial_thread_pools_){
            for(auto& itp_comp : initial_thread_pools_){
                if(itp.get_name() != itp_comp.get_name()){
                    if(threads::any(itp.get_pus() & itp_comp.get_pus())){
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // Returns true if any of the pools defined by the user is empty of resources
    // called in set_default_pool()
    bool resource_partitioner::check_empty_pools() const
    {
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for(size_t i(1); i<num_thread_pools; i++){
            if(!threads::any(initial_thread_pools_[i].get_pus())){
                return true;
            }
        }

        return false;
    }

    void resource_partitioner::set_threadmanager(threads::threadmanager_base* thrd_manag)
    {
        thread_manager_ = thrd_manag;
    }

    // create a new thread_pool, add it to the RP and return a pointer to it
    void resource_partitioner::create_thread_pool(const std::string &name, scheduling_policy sched)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

        if(name == "default")
            throw std::invalid_argument("cannot instantiate a initial_thread_pool named \"default\". The default pool is instantiated automatically by the resource-partitioner");

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for(size_t i(1); i<num_thread_pools; i++) {
            if(name == initial_thread_pools_[i].get_name())
                throw std::invalid_argument("there already exists a pool named " + name + ".\n");
        }

        initial_thread_pools_.push_back(init_pool_data(name, sched));
    }

    // ----------------------------------------------------------------------
    // Add processing units to pools via pu/core/domain api
    // ----------------------------------------------------------------------
    void resource_partitioner::add_resource(const pu &p,
        const std::string &pool_name)
    {
        get_pool(pool_name)->add_resource(p.id_);
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource(const std::vector<pu> &pv,
        const std::string &pool_name)
    {
        for (const pu &p : pv) {
            add_resource(p, pool_name);
        }
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource(const core &c,
        const std::string &pool_name)
    {
        add_resource(c.pus_, pool_name);
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource(const std::vector<core> &cv,
        const std::string &pool_name)
    {
        for (const core &c : cv) {
            add_resource(c.pus_, pool_name);
        }
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource(const numa_domain &nd,
        const std::string &pool_name)
    {
        add_resource(nd.cores_, pool_name);
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource(const std::vector<numa_domain> &ndv, const std::string &pool_name)
    {
        for (const numa_domain &d : ndv) {
            add_resource(d, pool_name);
        }
        set_affinity_from_resource_partitioner_ = true;
    }

    void resource_partitioner::add_resource_to_default(pu resource){
        add_resource(resource, "default");
        set_affinity_from_resource_partitioner_ = true;
    }

    // ----------------------------------------------------------------------
    //
    // ----------------------------------------------------------------------
    void resource_partitioner::set_scheduler(scheduling_policy sched, const std::string &pool_name){
        get_pool(pool_name)->set_scheduler(sched);
    }

    void resource_partitioner::init_rp(){

        //! FIXME
        //! copy all the little setups done in hpx_init
        //! and stick the here
        //! (this should be private)

    }

    ////////////////////////////////////////////////////////////////////////

    // this function is called in the constructor of thread_pool
    // returns a scheduler (moved) that thread pool should have as a data member
    scheduling_policy resource_partitioner::which_scheduler(const std::string &pool_name) {
        // look up which scheduler is needed
        scheduling_policy sched_type = get_pool(pool_name)->get_scheduling_policy();
        if(sched_type == unspecified)
            throw std::invalid_argument("Thread pool " + pool_name + " cannot be instantiated with unspecified scheduler type.");

        return sched_type;
    }

    threads::topology& resource_partitioner::get_topology() const
    {
        return topology_;
    }

    //! used in the constructor of runtime.
    //! but rt probs shouldn't even own one ...
    //! FIXME
    threads::policies::init_affinity_data resource_partitioner::get_init_affinity_data() const
    { //! should this return a pointer instead of a copy?
        return init_affinity_data_;
    }

    size_t resource_partitioner::get_num_pools() const{
        return initial_thread_pools_.size();
    }

    size_t resource_partitioner::get_num_threads(const std::string &pool_name)
    {
        return get_pool(pool_name)->get_num_threads();
    }

    const std::string &resource_partitioner::get_pool_name(size_t index) const {
        if(index >= initial_thread_pools_.size())
            throw std::invalid_argument("pool requested out of bounds.");
                    //! FIXME more detailed error message:
                    /*"pool " + i + " (zero-based index) requested out of bounds. "
                    "The resource_partitioner owns only " + initial_thread_pools_.size()
                    + " pools\n");*/
        return initial_thread_pools_[index].get_name();
    }

    threads::mask_cref_type resource_partitioner::get_pu_mask(std::size_t num_thread, bool numa_sensitive) const
    {
        return affinity_data_.get_pu_mask(num_thread, numa_sensitive, topology_);
    }


    ////////////////////////////////////////////////////////////////////////

    std::size_t resource_partitioner::get_pool_index(const std::string &pool_name) const {
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
    init_pool_data* resource_partitioner::get_pool(const std::string &pool_name) {
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
        //! FIXME Add names of available pools?
    }

    init_pool_data* resource_partitioner::get_default_pool() {
        auto pool = std::find_if(
                initial_thread_pools_.begin(), initial_thread_pools_.end(),
                [](init_pool_data itp) -> bool {return (itp.get_name() == "default");}
        );

        if(pool != initial_thread_pools_.end()){
            init_pool_data* ret(&(*pool)); //! FIXME yuck
            return ret;
        }

        throw std::invalid_argument(
                "the resource partitioner does not own a default pool \n");
    }

    void resource_partitioner::print_init_pool_data() const {           //! make this prettier
        std::cout << "the resource partitioner owns "
                  << initial_thread_pools_.size() << " pool(s) : \n";
        for(auto itp : initial_thread_pools_){
            itp.print_pool();
        }
    }

    ////////////////////////////////////////////////////////////////////////

    boost::atomic<int> resource_partitioner::instance_number_counter_(-1);

    } // namespace resource

} // namespace hpx
