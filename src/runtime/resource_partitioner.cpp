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

    resource::resource_partitioner & get_resource_partitioner()
    {
        util::static_<resource::resource_partitioner, std::false_type> rp;

        if(rp.get().cmd_line_parsed() == false){
            // if the resource partitioner is not accessed for the first time
            // if the command-line parsing has not yet been done
            throw std::invalid_argument("hpx::get_resource_partitioner() can be called only after the resource partitioner has been allowed to parse the command line options. Please call hpx::get_resource_partitioner(desc_cmdline, argc, argv) or hpx::get_resource_partitioner(argc, argv) instead");
            //! FIXME pas beau
        }

        return rp.get();
    }

    resource::resource_partitioner & get_resource_partitioner(int argc, char **argv)
    {
        using boost::program_options::options_description;

        options_description desc_cmdline(
                std::string("Usage: ") + HPX_APPLICATION_STRING +  " [options]");

        return get_resource_partitioner(desc_cmdline, argc, argv);
    }

    resource::resource_partitioner & get_resource_partitioner(
            boost::program_options::options_description desc_cmdline, int argc, char **argv)
    {
        util::static_<resource::resource_partitioner, std::false_type> rp_;
        auto& rp = rp_.get();

        if(rp.cmd_line_parsed() == false){ //! FIXME chnge to true and switch pos of instructions
            rp.parse(desc_cmdline, argc, argv);
            return rp;
        }

        throw std::invalid_argument("After hpx::get_resource_partitioner(desc_cmdline, argc, argv) or hpx::get_resource_partitioner(argc, argv) has been called already, you should call hpx::get_resource_partitioner()");
        //! FIXME pas beau
    }

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

    std::size_t init_pool_data::num_threads_overall = 0;

    init_pool_data::init_pool_data(const std::string &name, scheduling_policy sched)
        : pool_name_(name),
          scheduling_policy_(sched),
          num_threads_(0)
    {
        if (name.empty())
            throw std::invalid_argument(
                "cannot instantiate a thread_pool with empty string as a name.");
        // @TODO, remove unnecessary checks
    }

    init_pool_data::init_pool_data(const std::string &name,
        scheduler_function create_func)
        : pool_name_(name)
        , scheduling_policy_(user_defined)
        , num_threads_(0)
        , create_function_(create_func)
    {
        if (name.empty())
            throw std::invalid_argument(
                "cannot instantiate a thread pool with empty string as a name.");
    }

    // mechanism for adding resources
    // num threads = number of threads desired on a PU. defaults to 1.
    // note: if num_threads > 1 => oversubscription
    void init_pool_data::add_resource(std::size_t pu_index, std::size_t num_threads){

        if (pu_index >= hpx::threads::hardware_concurrency()) {
            throw std::invalid_argument("Processing unit index out of bounds.");
            //! FIXME give actual number of PUs
        }

        // Increment thread_num count (for pool-count and global count)
        num_threads_ += num_threads;
        num_threads_overall += num_threads;

        // Add pu mask to internal data structure
        threads::mask_type pu_mask = 0;
        threads::set(pu_mask, pu_index);
        assigned_pus_.push_back(pu_mask);
    }

    void init_pool_data::print_pool() const {
        std::cout << "[pool \"" << pool_name_ << "\"] with scheduler " ;
        std::string sched;
        switch(scheduling_policy_) {
            case -1 : sched = "unspecified";        break;
            case -2 : sched = "user supplied";      break;
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
        for(size_t i(0); i<assigned_pus_.size(); i++)
            std::cout << bitset_type(assigned_pus_[i]) << "\n";
    }


    ////////////////////////////////////////////////////////////////////////

    resource_partitioner::resource_partitioner()
            : thread_manager_(nullptr),
              topology_(threads::create_topology())
    {
        // Create the default pool
        initial_thread_pools_.push_back(init_pool_data("default"));

        // allow only one resource_partitioner instance
        if(instance_number_counter_++ >= 0)
            throw std::runtime_error("Cannot instantiate more than one resource partitioner");
    }

    void resource_partitioner::set_hpx_init_options(
            hpx::runtime_mode mode,
            util::function_nonser<
                    int(boost::program_options::variables_map& vm)
            > const& f,
            std::vector<std::string> && ini_config)
    {
        cfg_.rtcfg_.mode_ = mode;
        cfg_.hpx_main_f_ = f;
        cfg_.ini_config_ = std::move(ini_config);
    }

    int resource_partitioner::call_cmd_line_options(
            boost::program_options::options_description const& desc_cmdline,
            int argc, char** argv)
    {
        return cfg_.call(desc_cmdline, argc, argv);
    }

    bool resource_partitioner::pu_exposed(std::size_t pid)
    {
        threads::mask_type pu_mask(0);
        threads::set(pu_mask, pid);
        threads::mask_type comp = affinity_data_.get_used_pus_mask();
        return (comp & pu_mask);
    }

    void resource_partitioner::fill_topology_vectors()
    {
        std::size_t pid = 0;
        std::size_t N = topology_.get_number_of_numa_nodes();
        if (N==0) N= topology_.get_number_of_sockets();
        numa_domains_.reserve(N);

        // loop on the numa-domains
        for (std::size_t i=0; i<N; ++i) {
            numa_domains_.push_back(numa_domain()); // add a numa domain
            numa_domain &nd = numa_domains_.back(); // get a handle to the numa-domain just added
            nd.id_          = i;                    // set its index
            nd.cores_.reserve(topology_.get_number_of_numa_node_cores(i));

            bool numa_domain_contains_exposed_cores = false;

            // loop on the cores
            for (std::size_t j=0; j<topology_.get_number_of_numa_node_cores(i); ++j) {
                nd.cores_.push_back(core());
                core &c   = nd.cores_.back();
                c.id_     = j;
                c.domain_ = &nd;
                c.pus_.reserve(topology_.get_number_of_core_pus(j));

                bool core_contains_exposed_pus = false;

                // loop on the processing units
                for (std::size_t k=0; k<topology_.get_number_of_core_pus(j); ++k) {
                    if(pu_exposed(pid)){
                        c.pus_.push_back(pu());
                        pu &p   = c.pus_.back();

                        p.id_   = pid;
                        p.core_ = &c;

                        p.thread_occupancy_ = affinity_data_.get_thread_occupancy(pid);
                        if(p.thread_occupancy_ == 0)
                            throw std::runtime_error("PU #" + std::to_string(pid) + " has thread occupancy 0");
                        p.thread_occupancy_count = p.thread_occupancy_;
                        core_contains_exposed_pus = true;
                    }

                    pid++;

                }

                if(core_contains_exposed_pus){
                    numa_domain_contains_exposed_cores = true;
                } else {
                    nd.cores_.pop_back();
                }

            }

            if(!numa_domain_contains_exposed_cores)
                nd.cores_.pop_back();

        }
    }

    // This function is called in hpx_init, before the instantiation of the runtime
    // it takes care of all the initialization of the initial pool data held by the resource partitioner
    // -1 checks whether there are empty pools
    // -2 checks whether there are oversubscribed PUs
    // -3 sets data member thread_num for each pool
    // -4 sets up the default pool
    //! FIXME update these comments
    void resource_partitioner::setup_pools() {

        // check whether any of the pools defined up to now are empty
        // note: does not check "default", this one is allowed not to be given resources by the user
        if(check_empty_pools()) {
            throw std::invalid_argument("Pools empty of resources are not allowed. Please re-run this program with allow-empty-pool-policy (not implemented yet)");
        }
        //! FIXME add allow-empty-pools policy. Wait, does this even make sense??

        // Assign all free resources to the default pool
        for (hpx::resource::numa_domain &d : numa_domains_) {
            for (hpx::resource::core &c : d.cores_) {
                for (hpx::resource::pu &p : c.pus_) {
                    std::size_t threads_to_add = p.thread_occupancy_ - p.thread_occupancy_count;
                    if (threads_to_add > 0) {
                        add_resource(p, "default", threads_to_add);
                    }
                }
            }
        }

        // make sure the mask for the default pool has resources in it
        //! FIXME add allow-empty-default-pool policy

    }

    void resource_partitioner::setup_schedulers() {

        // select the default scheduler
        scheduling_policy default_scheduler;

        if (0 == std::string("local").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::local;
        }
        else if (0 == std::string("local-priority-fifo").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::local_priority_fifo ;
        }
        else if (0 == std::string("local-priority-lifo").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::local_priority_lifo;
        }
        else if (0 == std::string("static").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::static_;
        }
        else if (0 == std::string("static-priority").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::static_priority;
        }
        else if (0 == std::string("abp-priority").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::abp_priority;
        }
        else if (0 == std::string("hierarchy").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::hierarchy;
        }
        else if (0 == std::string("periodic-priority").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::periodic_priority;
        }
        else if (0 == std::string("throttle").find(cfg_.queuing_)) {
            default_scheduler = scheduling_policy::throttle;
        }
        else {
            throw detail::command_line_error(
                    "Bad value for command line option --hpx:queuing");
        }

        // set this scheduler on the pools that do not have a specified scheduler yet
        std::size_t npools(initial_thread_pools_.size());
        for(size_t i(0); i<npools; i++){
            if(initial_thread_pools_[i].scheduling_policy_ == unspecified){
                initial_thread_pools_[i].scheduling_policy_ = default_scheduler;
            }
        }
    }

    // Returns true if any of the pools defined by the user is empty of resources
    // called in set_default_pool()
    bool resource_partitioner::check_empty_pools() const
    {
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for(size_t i(1); i<num_thread_pools; i++){
            for(auto assigned_pus : initial_thread_pools_[i].assigned_pus_){
                if(!threads::any(assigned_pus)) {
                    return true;
                }
            }
        }

        return false;
    }

    void resource_partitioner::set_threadmanager(threads::threadmanager_base* thrd_manag)
    {
        thread_manager_ = thrd_manag;
    }

    // create a new thread_pool
    void resource_partitioner::create_thread_pool(const std::string &name, scheduling_policy sched)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

        if (name == "default") {
            initial_thread_pools_[0] = init_pool_data("default", sched);
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for(size_t i(1); i<num_thread_pools; i++) {
            if(name == initial_thread_pools_[i].pool_name_)
                throw std::invalid_argument("there already exists a pool named " + name + ".\n");
        }

        initial_thread_pools_.push_back(init_pool_data(name, sched));
    }

    // create a new thread_pool
    void resource_partitioner::create_thread_pool(const std::string &name, scheduler_function scheduler_creation)
    {
        if(name.empty())
            throw std::invalid_argument("cannot instantiate a initial_thread_pool with empty string as a name.");

        if (name == "default") {
            initial_thread_pools_[0] = init_pool_data("default", scheduler_creation);
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for(size_t i(1); i<num_thread_pools; i++) {
            if(name == initial_thread_pools_[i].pool_name_)
                throw std::invalid_argument("there already exists a pool named " + name + ".\n");
        }

        initial_thread_pools_.push_back(init_pool_data(name, scheduler_creation));
    }

    // ----------------------------------------------------------------------
    // Add processing units to pools via pu/core/domain api
    // ----------------------------------------------------------------------
    void resource_partitioner::add_resource(pu &p,
        const std::string &pool_name, std::size_t num_threads)
    {
        //! FIXME except if policy allow_extra_thread_creation is activated
        //! then I don't have to check the occupancy count ...
        // check occupancy counter and decrement it
        if (p.thread_occupancy_count > 0){
            p.thread_occupancy_count--;
            get_pool(pool_name)->add_resource(p.id_, num_threads);

            // Make sure the total number of requested threads does not exceed the number of threads
            // requested on the command line
            //! FIXME except if policy allow_extra_thread_creation is activated
            if(init_pool_data::num_threads_overall >= cfg_.num_threads_){
                //! FIXME add allow_empty_default_pool policy
/*                if(rp-policy == allow_empty_default_pool
                    && init_pool_data::num_threads_overall == cfg_.num_threads_){
                    // then it's all fine
                } else {*/
                    throw std::runtime_error("Creation of " +
                        std::to_string(init_pool_data::num_threads_overall) + " threads requested by " +
                        "the resource partitioner, but only " + std::to_string(cfg_.num_threads_) +
                        " provided in the command-line.");
//                }
            }
        } else {
            throw std::runtime_error("PU #" + std::to_string(p.id_) + " can be assigned only " +
            std::to_string(p.thread_occupancy_) + " threads according to affinity bindings.");
        }
    }

    void resource_partitioner::add_resource(std::vector<pu> &pv,
        const std::string &pool_name)
    {
        for (pu &p : pv) {
            add_resource(p, pool_name);
        }
    }

    void resource_partitioner::add_resource(core &c,
        const std::string &pool_name)
    {
        add_resource(c.pus_, pool_name);
    }

    void resource_partitioner::add_resource(std::vector<core> &cv,
        const std::string &pool_name)
    {
        for (core &c : cv) {
            add_resource(c.pus_, pool_name);
        }
    }

    void resource_partitioner::add_resource(numa_domain &nd,
        const std::string &pool_name)
    {
        add_resource(nd.cores_, pool_name);
    }

    void resource_partitioner::add_resource(std::vector<numa_domain> &ndv, const std::string &pool_name)
    {
        for (numa_domain &d : ndv) {
            add_resource(d, pool_name);
        }
    }

    // ----------------------------------------------------------------------
    //
    // ----------------------------------------------------------------------
    void resource_partitioner::set_scheduler(scheduling_policy sched, const std::string &pool_name){
        get_pool(pool_name)->scheduling_policy_ = sched;
    }

    void resource_partitioner::configure_pools(){
        setup_pools();
        setup_schedulers();
    }

    ////////////////////////////////////////////////////////////////////////

    // this function is called in the constructor of thread_pool
    // returns a scheduler (moved) that thread pool should have as a data member
    scheduling_policy resource_partitioner::which_scheduler(const std::string &pool_name) {
        // look up which scheduler is needed
        scheduling_policy sched_type = get_pool(pool_name)->scheduling_policy_;
        if(sched_type == unspecified)
            throw std::invalid_argument("Thread pool " + pool_name + " cannot be instantiated with unspecified scheduler type.");

        return sched_type;
    }

    threads::topology& resource_partitioner::get_topology() const
    {
        return topology_;
    }

    util::command_line_handling& resource_partitioner::get_command_line_switches()
    {
        return cfg_;
    }

    std::size_t resource_partitioner::get_num_threads() const
    {
        return cfg_.num_threads_;
    }

    size_t resource_partitioner::get_num_pools() const{
        return initial_thread_pools_.size();
    }

    size_t resource_partitioner::get_num_threads(const std::string &pool_name)
    {
        return get_pool(pool_name)->num_threads_;
    }

    const std::string &resource_partitioner::get_pool_name(size_t index) const {
        if(index >= initial_thread_pools_.size())
            throw std::invalid_argument("pool requested out of bounds.");
                    //! FIXME more detailed error message:
                    /*"pool " + i + " (zero-based index) requested out of bounds. "
                    "The resource_partitioner owns only " + initial_thread_pools_.size()
                    + " pools\n");*/
        return initial_thread_pools_[index].pool_name_;
    }

    threads::mask_cref_type resource_partitioner::get_pu_mask(std::size_t num_thread, bool numa_sensitive) const
    {
        return affinity_data_.get_pu_mask(num_thread, numa_sensitive);
    }

    bool resource_partitioner::cmd_line_parsed() const
    {
        return (cfg_.parsed_ == true);
    }

    void resource_partitioner::parse(boost::program_options::options_description desc_cmdline, int argc, char **argv)
    {
        // set internal parameters of runtime configuration
        cfg_.rtcfg_ = util::runtime_configuration(argv[0]);

        // parse command line and set options
        cfg_.call(desc_cmdline, argc, argv);

        // set all parameters related to affinity data
        cores_needed_ = affinity_data_.init(cfg_);
        //! FIXME what is this?? Change name ...

        // set data describing internal topology backend
        fill_topology_vectors();
    }

    const scheduler_function &resource_partitioner::get_pool_creator(size_t index) const {
        if (index >= initial_thread_pools_.size()) {
            throw std::invalid_argument("pool requested out of bounds.");
        }
        return initial_thread_pools_[index].create_function_;
    }

    ////////////////////////////////////////////////////////////////////////

    std::size_t resource_partitioner::get_pool_index(const std::string &pool_name) const {
        std::size_t N = initial_thread_pools_.size();
        for(size_t i(0); i<N; i++) {
            if (initial_thread_pools_[i].pool_name_ == pool_name) {
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
                [&pool_name](init_pool_data itp) -> bool {return (itp.pool_name_ == pool_name);}
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
                [](init_pool_data itp) -> bool {return (itp.pool_name_ == "default");}
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
