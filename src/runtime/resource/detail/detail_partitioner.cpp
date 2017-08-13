//  Copyright (c)      2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/resource/detail/partitioner.hpp>
#include <hpx/runtime/resource/partitioner.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/runtime/threads/detail/thread_pool_base.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/static.hpp>

#include <boost/atomic.hpp>

#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace hpx { namespace resource { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_ATTRIBUTE_NORETURN void throw_runtime_error(
        std::string const& func, std::string const& message)
    {
        if (get_runtime_ptr() != nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status, func, message);
        }
        else
        {
            throw std::runtime_error(func + ": " +  message);
        }
    }

    HPX_ATTRIBUTE_NORETURN void throw_invalid_argument(
        std::string const& func, std::string const& message)
    {
        if (get_runtime_ptr() != nullptr)
        {
            HPX_THROW_EXCEPTION(bad_parameter, func, message);
        }
        else
        {
            throw std::invalid_argument(func + ": " +  message);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::size_t init_pool_data::num_threads_overall = 0;

    init_pool_data::init_pool_data(
            std::string const& name, scheduling_policy sched)
        : pool_name_(name)
        , scheduling_policy_(sched)
        , num_threads_(0)
    {
        if (name.empty())
        {
            throw_invalid_argument("init_pool_data::init_pool_data",
                "cannot instantiate a thread_pool with empty string as a name.");
        }
    }

    init_pool_data::init_pool_data(std::string const& name,
            scheduler_function create_func)
        : pool_name_(name)
        , scheduling_policy_(user_defined)
        , num_threads_(0)
        , create_function_(std::move(create_func))
    {
        if (name.empty())
        {
            throw_invalid_argument("init_pool_data::init_pool_data",
                    "cannot instantiate a thread_pool with empty string "
                    "as a name.");
        }
    }

    // mechanism for adding resources
    // num threads = number of threads desired on a PU. defaults to 1.
    // note: if num_threads > 1 => oversubscription
    void init_pool_data::add_resource(
        std::size_t pu_index, bool exclusive, std::size_t num_threads)
    {
        if (pu_index >= hpx::threads::hardware_concurrency())
        {
            throw_invalid_argument("init_pool_data::add_resource",
                    "init_pool_data::add_resource: processing unit index "
                    "out of bounds. The total available number of "
                    "processing units on this machine is " +
                    std::to_string(hpx::threads::hardware_concurrency()));
        }

        // Increment thread_num count (for pool-count and global count)
        num_threads_ += num_threads;
        num_threads_overall += num_threads;

        // Add pu mask to internal data structure
        threads::mask_type pu_mask = threads::mask_type();
        threads::resize(pu_mask, threads::hardware_concurrency());
        threads::set(pu_mask, pu_index);

        // Add one mask for each OS-thread
        for (std::size_t i = 0; i != num_threads; i++)
        {
            assigned_pus_.push_back(pu_mask);
            assigned_pu_nums_.push_back(
                util::make_tuple(pu_index, exclusive, false)
            );
        }
    }

    void init_pool_data::print_pool(std::ostream& os) const
    {
        os << "[pool \"" << pool_name_ << "\"] with scheduler ";

        std::string sched;
        switch (scheduling_policy_)
        {
        case resource::unspecified:
            sched = "unspecified";
            break;
        case resource::user_defined:
            sched = "user supplied";
            break;
        case resource::local:
            sched = "local";
            break;
        case resource::local_priority_fifo:
            sched = "local_priority_fifo";
            break;
        case resource::local_priority_lifo:
            sched = "local_priority_lifo";
            break;
        case resource::static_:
            sched = "static";
            break;
        case resource::static_priority:
            sched = "static_priority";
            break;
        case resource::abp_priority:
            sched = "abp_priority";
            break;
        case resource::hierarchy:
            sched = "hierarchy";
            break;
        case resource::periodic_priority:
            sched = "periodic_priority";
            break;
        case resource::throttle:
            sched = "throttle";
            break;
        }

        os << "\"" << sched << "\" is running on PUs : \n";

        for (threads::mask_cref_type assigned_pu : assigned_pus_)
        {
            os << std::hex << HPX_CPU_MASK_PREFIX << assigned_pu << '\n';
        }
    }

    void init_pool_data::assign_pu(std::size_t virt_core)
    {
        HPX_ASSERT(virt_core <= assigned_pu_nums_.size());
        HPX_ASSERT(!util::get<2>(assigned_pu_nums_[virt_core]));

        util::get<2>(assigned_pu_nums_[virt_core]) = true;
    }

    void init_pool_data::unassign_pu(std::size_t virt_core)
    {
        HPX_ASSERT(virt_core <= assigned_pu_nums_.size());
        HPX_ASSERT(util::get<2>(assigned_pu_nums_[virt_core]));

        util::get<2>(assigned_pu_nums_[virt_core]) = false;
    }

    bool init_pool_data::pu_is_exclusive(std::size_t virt_core) const
    {
        HPX_ASSERT(virt_core <= assigned_pu_nums_.size());
        HPX_ASSERT(util::get<2>(assigned_pu_nums_[virt_core]));

        return util::get<1>(assigned_pu_nums_[virt_core]);
    }

    bool init_pool_data::pu_is_assigned(std::size_t virt_core) const
    {
        HPX_ASSERT(virt_core <= assigned_pu_nums_.size());
        HPX_ASSERT(util::get<2>(assigned_pu_nums_[virt_core]));

        return util::get<2>(assigned_pu_nums_[virt_core]);
    }

    // 'shift' all thread assignments up by the first_core offset
    void init_pool_data::assign_first_core(std::size_t first_core)
    {
        for (std::size_t i = 0; i != num_threads_; ++i)
        {
             std::size_t& pu_num = util::get<0>(assigned_pu_nums_[i]);
             pu_num = (pu_num + first_core) % threads::hardware_concurrency();

             threads::reset(assigned_pus_[i]);
             threads::set(assigned_pus_[i], pu_num);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    partitioner::partitioner()
        : first_core_(std::size_t(-1))
        , cores_needed_(std::size_t(-1))
        , topology_(threads::create_topology())
        , mode_(mode_default)
    {
        // allow only one partitioner instance
        if (++instance_number_counter_ > 1)
        {
            throw_runtime_error("partitioner::partitioner",
                "Cannot instantiate more than one resource partitioner");
        }

        // Create the default pool
        initial_thread_pools_.push_back(init_pool_data("default"));
    }

    bool partitioner::pu_exposed(std::size_t pu_num)
    {
        threads::mask_type pu_mask = threads::mask_type();
        threads::set(pu_mask, pu_num);

        threads::mask_type comp = affinity_data_.get_used_pus_mask(pu_num);
        return threads::any(comp & pu_mask);
    }

    void partitioner::fill_topology_vectors()
    {
        std::size_t pid = 0;
        std::size_t num_numa_nodes = topology_.get_number_of_numa_nodes();
        if (num_numa_nodes == 0)
            num_numa_nodes = topology_.get_number_of_sockets();
        numa_domains_.reserve(num_numa_nodes);

        // loop on the numa-domains
        for (std::size_t i = 0; i != num_numa_nodes; ++i)
        {
            numa_domains_.emplace_back(i);    // add a numa domain
            numa_domain &nd = numa_domains_.back();     // numa-domain just added

            std::size_t numa_node_cores =
                topology_.get_number_of_numa_node_cores(i);
            nd.cores_.reserve(numa_node_cores);

            bool numa_domain_contains_exposed_cores = false;

            // loop on the cores
            for (std::size_t j = 0; j != numa_node_cores; ++j)
            {
                nd.cores_.emplace_back(j, &nd);
                core &c = nd.cores_.back();

                std::size_t core_pus = topology_.get_number_of_core_pus(j);
                c.pus_.reserve(core_pus);

                bool core_contains_exposed_pus = false;

                // loop on the processing units
                for (std::size_t k = 0; k != core_pus; ++k)
                {
                    if (pu_exposed(pid))
                    {
                        c.pus_.emplace_back(pid, &c,
                            affinity_data_.get_thread_occupancy(pid));
                        pu &p = c.pus_.back();

                        if (p.thread_occupancy_ == 0)
                        {
                            throw_runtime_error(
                                "partitioner::fill_topology_vectors",
                                "PU #" + std::to_string(pid) +
                                " has thread occupancy 0");
                        }
                        core_contains_exposed_pus = true;
                    }
                    ++pid;
                }

                if (core_contains_exposed_pus)
                {
                    numa_domain_contains_exposed_cores = true;
                }
                else
                {
                    nd.cores_.pop_back();
                }
            }

            if (!numa_domain_contains_exposed_cores)
            {
                numa_domains_.pop_back();
            }
        }
    }

    std::size_t partitioner::assign_cores(std::size_t first_core)
    {
        std::lock_guard<mutex_type> l(mtx_);

        // adjust first_core, if needed
        if (first_core_ != first_core)
        {
            std::size_t offset = first_core;
            std::size_t num_pus_core = topology_.get_number_of_core_pus(offset);

            if (first_core_ != std::size_t(-1))
            {
                offset -= first_core_;
            }

            if (offset != 0)
            {
                offset *= num_pus_core;
                for (auto& d : initial_thread_pools_)
                {
                    d.assign_first_core(offset);
                }
            }
            first_core_ = first_core;
            reconfigure_affinities_locked();
        }

        // should have been initialized by now
        HPX_ASSERT(cores_needed_ != std::size_t(-1));
        return cores_needed_;
    }

    // This function is called in hpx_init, before the instantiation of the
    // runtime It takes care of configuring some internal parameters of the
    // resource partitioner related to the pools
    // -1 assigns all free resources to the default pool
    // -2 checks whether there are empty pools
    void partitioner::setup_pools()
    {
        // Assign all free resources to the default pool
        bool first = true;
        for (hpx::resource::numa_domain &d : numa_domains_)
        {
            for (hpx::resource::core &c : d.cores_)
            {
                for (hpx::resource::pu &p : c.pus_)
                {
                    if (p.thread_occupancy_count_ == 0)
                    {
                        // The default pool resources are assigned non-
                        // exclusively if dynamic pools are enabled.
                        // Also, by default, the first PU is always exclusive
                        // (to avoid deadlocks).
                        add_resource(p, "default",
                            first || !(mode_ & mode_allow_dynamic_pools));
                        first = false;
                    }
                }
            }
        }

        std::lock_guard<mutex_type> l(mtx_);

        // @TODO allow empty pools
        if (get_pool_data("default").num_threads_ == 0)
        {
            throw_runtime_error("partitioner::setup_pools",
                "Default pool has no threads assigned. Please rerun with "
                "--hpx:threads=X and check the pool thread assignment");
        }

        // Check whether any of the pools defined up to now are empty
        if (check_empty_pools())
        {
            throw_runtime_error("partitioner::setup_pools",
                "Pools empty of resources are not allowed. Please re-run this "
                "application with allow-empty-pool-policy (not implemented "
                "yet)");
        }
        //! FIXME add allow-empty-pools policy. Wait, does this even make sense??
    }

    // This function is called in hpx_init, before the instantiation of the runtime
    // It takes care of configuring some internal parameters of the resource partitioner
    // related to the pools' schedulers
    void partitioner::setup_schedulers()
    {
        // select the default scheduler
        scheduling_policy default_scheduler;

        if (0 == std::string("local").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::local;
        }
        else if (0 == std::string("local-priority-fifo").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::local_priority_fifo;
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
        else if (0 == std::string("throttle").find(cfg_.queuing_))
        {
            default_scheduler = scheduling_policy::throttle;
        }
        else
        {
            throw hpx::detail::command_line_error(
                "Bad value for command line option --hpx:queuing");
        }

        // set this scheduler on the pools that do not have a specified scheduler yet
        std::lock_guard<mutex_type> l(mtx_);
        std::size_t npools = initial_thread_pools_.size();
        for (std::size_t i = 0; i != npools; ++i)
        {
            if (initial_thread_pools_[i].scheduling_policy_ == unspecified)
            {
                initial_thread_pools_[i].scheduling_policy_ = default_scheduler;
            }
        }
    }

    // This function is called in hpx_init, before the instantiation of the
    // runtime. It takes care of configuring some internal parameters of the
    // resource partitioner related to the affinity bindings
    //
    // If we use the resource partitioner, OS-thread numbering gets slightly
    // complicated: The affinity_masks_ data member of affinity_data considers
    // OS-threads to be numbered in order of occupation of the consecutive
    // processing units, while the thread manager will consider them to be
    // ordered according to their assignment to pools (first all threads
    // belonging to the default pool, then all threads belonging to the first
    // pool created, etc.) and instantiate them according to this system.
    // We need to re-write affinity_data_ with the masks in the correct order
    // at this stage.
    void partitioner::reconfigure_affinities()
    {
        std::lock_guard<mutex_type> l(mtx_);
        reconfigure_affinities_locked();
    }

    void partitioner::reconfigure_affinities_locked()
    {
        std::vector<std::size_t> new_pu_nums;
        std::vector<threads::mask_type> new_affinity_masks;

        new_pu_nums.reserve(initial_thread_pools_.size());
        new_affinity_masks.reserve(initial_thread_pools_.size());

        {
            for (auto &itp : initial_thread_pools_)
            {
                for (auto const& mask : itp.assigned_pus_)
                {
                    new_affinity_masks.push_back(mask);
                }
                for (auto const& pu_num : itp.assigned_pu_nums_)
                {
                    new_pu_nums.push_back(util::get<0>(pu_num));
                }
            }
        }

        affinity_data_.set_num_threads(new_pu_nums.size());
        affinity_data_.set_pu_nums(std::move(new_pu_nums));
        affinity_data_.set_affinity_masks(std::move(new_affinity_masks));
        affinity_data_.init_cached_pu_nums(new_pu_nums.size());
    }

    // Returns true if any of the pools defined by the user is empty of resources
    // called in set_default_pool()
    bool partitioner::check_empty_pools() const
    {
        std::size_t num_thread_pools = initial_thread_pools_.size();

        for (std::size_t i = 0; i != num_thread_pools; i++)
        {
            if (initial_thread_pools_[i].assigned_pus_.empty())
            {
                return false;
            }
            for (auto assigned_pus : initial_thread_pools_[i].assigned_pus_)
            {
                if (!threads::any(assigned_pus))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // create a new thread_pool
    void partitioner::create_thread_pool(
        std::string const& pool_name, scheduling_policy sched)
    {
        if (get_runtime_ptr() != nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "partitioner::create_thread_pool",
                "this function must be called before the runtime system has "
                "been started");
        }

        if (pool_name.empty())
        {
            throw std::invalid_argument(
                "partitioner::create_thread_pool: "
                "cannot instantiate a initial_thread_pool with empty string "
                "as a name.");
        }

        std::lock_guard<mutex_type> l(mtx_);

        if (pool_name == "default")
        {
            initial_thread_pools_[0] = detail::init_pool_data("default", sched);
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for (std::size_t i = 1; i < num_thread_pools; i++)
        {
            if (pool_name == initial_thread_pools_[i].pool_name_)
            {
                throw std::invalid_argument(
                    "partitioner::create_thread_pool: "
                    "there already exists a pool named '" + pool_name + "'.\n");
            }
        }

        initial_thread_pools_.push_back(detail::init_pool_data(pool_name, sched));
    }

    // create a new thread_pool
    void partitioner::create_thread_pool(
        std::string const& pool_name, scheduler_function scheduler_creation)
    {
        if (get_runtime_ptr() != nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "partitioner::create_thread_pool",
                "this function must be called before the runtime system has "
                "been started");
        }

        if (pool_name.empty())
        {
            throw std::invalid_argument(
                "partitioner::create_thread_pool: "
                "cannot instantiate a initial_thread_pool with empty string "
                "as a name.");
        }

        std::lock_guard<mutex_type> l(mtx_);

        if (pool_name == "default")
        {
            initial_thread_pools_[0] = detail::init_pool_data(
                "default", std::move(scheduler_creation));
            return;
        }

        //! if there already exists a pool with this name
        std::size_t num_thread_pools = initial_thread_pools_.size();
        for (std::size_t i = 1; i != num_thread_pools; ++i)
        {
            if (pool_name == initial_thread_pools_[i].pool_name_)
            {
                throw std::invalid_argument(
                    "partitioner::create_thread_pool: "
                    "there already exists a pool named '" + pool_name + "'.\n");
            }
        }

        initial_thread_pools_.push_back(
            detail::init_pool_data(pool_name, std::move(scheduler_creation)));
    }

    // ----------------------------------------------------------------------
    // Add processing units to pools via pu/core/domain api
    // ----------------------------------------------------------------------
    void partitioner::add_resource(
        pu const& p, std::string const& pool_name, bool exclusive,
        std::size_t num_threads)
    {
        if (get_runtime_ptr() != nullptr)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "partitioner::add_resource",
                "this function must be called before the runtime system has "
                "been started");
        }

        std::lock_guard<mutex_type> l(mtx_);

        if (!exclusive && !(mode_ & mode_allow_dynamic_pools))
        {
            throw std::invalid_argument(
                "partitioner::add_resource: dynamic pools have not been "
                "enabled for this partitioner");
        }

        if (mode_ & mode_allow_oversubscription)
        {
            // increment occupancy counter
            get_pool_data(pool_name).add_resource(
                p.id_, exclusive, num_threads);
            ++p.thread_occupancy_count_;
            return;
        }

        // check occupancy counter and increment it
        if (p.thread_occupancy_count_ == 0)
        {
            get_pool_data(pool_name).add_resource(
                p.id_, exclusive, num_threads);
            ++p.thread_occupancy_count_;

            // Make sure the total number of requested threads does not exceed
            // the number of threads requested on the command line
            if (detail::init_pool_data::num_threads_overall > cfg_.num_threads_)
            {
                //! FIXME add allow_empty_default_pool policy
                /*                if (rp-policy == allow_empty_default_pool
                    && detail::init_pool_data::num_threads_overall == cfg_.num_threads_) {
                    // then it's all fine
                } else {*/
                throw std::runtime_error(
                    "partitioner::add_resource: " "Creation of " +
                    std::to_string(detail::init_pool_data::num_threads_overall) +
                        " threads requested by the resource partitioner, but "
                        "only " +
                    std::to_string(cfg_.num_threads_) +
                        " provided on the command-line.");
                //                }
            }
        }
        else
        {
            throw std::runtime_error(
                "partitioner::add_resource: " "PU #" + std::to_string(p.id_) +
                " can be assigned only " + std::to_string(p.thread_occupancy_) +
                " threads according to affinity bindings.");
        }
    }

    void partitioner::add_resource(std::vector<pu> const& pv,
        std::string const& pool_name, bool exclusive)
    {
        for (pu const& p : pv)
        {
            add_resource(p, pool_name, exclusive);
        }
    }

    void partitioner::add_resource(core const& c,
        std::string const& pool_name, bool exclusive)
    {
        add_resource(c.pus_, pool_name, exclusive);
    }

    void partitioner::add_resource(std::vector<core> const& cv,
        std::string const& pool_name, bool exclusive)
    {
        for (const core &c : cv)
        {
            add_resource(c.pus_, pool_name, exclusive);
        }
    }

    void partitioner::add_resource(numa_domain const& nd,
        std::string const& pool_name, bool exclusive)
    {
        add_resource(nd.cores_, pool_name, exclusive);
    }

    void partitioner::add_resource(
        std::vector<numa_domain> const& ndv,
        std::string const& pool_name, bool exclusive)
    {
        for (const numa_domain &d : ndv)
        {
            add_resource(d, pool_name, exclusive);
        }
    }

    void partitioner::set_scheduler(
        scheduling_policy sched, std::string const& pool_name)
    {
        std::lock_guard<mutex_type> l(mtx_);
        get_pool_data(pool_name).scheduling_policy_ = sched;
    }

    void partitioner::configure_pools()
    {
        setup_pools();
        setup_schedulers();
        reconfigure_affinities();
    }

    ////////////////////////////////////////////////////////////////////////
    // this function is called in the constructor of thread_pool
    // returns a scheduler (moved) that thread pool should have as a data member
    scheduling_policy partitioner::which_scheduler(
        std::string const& pool_name)
    {
        std::lock_guard<mutex_type> l(mtx_);

        // look up which scheduler is needed
        scheduling_policy sched_type =
            get_pool_data(pool_name).scheduling_policy_;
        if (sched_type == unspecified)
        {
            throw std::invalid_argument(
                "partitioner::which_scheduler: " "Thread pool " + pool_name +
                " cannot be instantiated with unspecified scheduler type.");
        }
        return sched_type;
    }

    threads::topology &partitioner::get_topology() const
    {
        return topology_;
    }

    util::command_line_handling &
    partitioner::get_command_line_switches()
    {
        return cfg_;
    }

    std::size_t partitioner::get_num_distinct_pus() const
    {
        return cfg_.num_threads_;
    }

    std::size_t partitioner::get_num_threads() const
    {
        std::size_t num_threads = 0;

        {
            std::lock_guard<mutex_type> l(mtx_);
            std::size_t num_thread_pools = initial_thread_pools_.size();
            for (size_t i = 0; i != num_thread_pools; ++i)
            {
                num_threads += get_pool_data(i).num_threads_;
            }
        }

        // the number of allocated threads should be the same as the number of
        // threads to create (if no over-subscription is allowed)
        HPX_ASSERT(mode_ & mode_allow_oversubscription ||
            num_threads == cfg_.num_threads_);

        return num_threads;
    }

    std::size_t partitioner::get_num_pools() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return initial_thread_pools_.size();
    }

    std::size_t partitioner::get_num_threads(
        std::size_t pool_index) const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return get_pool_data(pool_index).num_threads_;
    }

    std::size_t partitioner::get_num_threads(
        const std::string &pool_name) const
    {
        std::lock_guard<mutex_type> l(mtx_);
        return get_pool_data(pool_name).num_threads_;
    }

    detail::init_pool_data const& partitioner::get_pool_data(
        std::size_t pool_index) const
    {
        if (pool_index >= initial_thread_pools_.size())
        {
            throw_invalid_argument(
                "partitioner::get_pool_data",
                "pool index " + std::to_string(pool_index) +
                    " too large: the resource partitioner owns only " +
                std::to_string(initial_thread_pools_.size()) +
                    " thread pools.");
        }
        return initial_thread_pools_[pool_index];
    }

    std::string const& partitioner::get_pool_name(
        std::size_t index) const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (index >= initial_thread_pools_.size())
        {
            throw_invalid_argument(
                "partitioner::get_pool_name: ",
                "pool " + std::to_string(index) +
                " (zero-based index) requested out of bounds. The "
                "partitioner owns only " +
                std::to_string(initial_thread_pools_.size()) + " pools");
        }
        return initial_thread_pools_[index].pool_name_;
    }

    size_t partitioner::get_pu_num(std::size_t global_thread_num)
    {
        return affinity_data_.get_pu_num(global_thread_num);
    }

    threads::mask_cref_type partitioner::get_pu_mask(
        std::size_t global_thread_num) const
    {
        return affinity_data_.get_pu_mask(global_thread_num);
    }

    bool partitioner::cmd_line_parsed() const
    {
        return (cfg_.cmd_line_parsed_ == true);
    }

    int partitioner::parse(
        util::function_nonser<
            int(boost::program_options::variables_map& vm)
        > const& f,
        boost::program_options::options_description desc_cmdline, int argc,
        char **argv, std::vector<std::string> ini_config,
        resource::partitioner_mode rpmode, runtime_mode mode,
        bool fill_internal_topology)
    {
        mode_ = rpmode;

        // set internal parameters of runtime configuration
        cfg_.rtcfg_ = util::runtime_configuration(argv[0], mode);
        cfg_.ini_config_ = std::move(ini_config);
        cfg_.hpx_main_f_ = f;

        // parse command line and set options
        // terminate set if program options contain --hpx:help or --hpx:version ...
        cfg_.parse_terminate_ = cfg_.call(desc_cmdline, argc, argv);

        // set all parameters related to affinity data
        cores_needed_ = affinity_data_.init(cfg_);

        if (fill_internal_topology)
        {
            // set data describing internal topology back-end
            fill_topology_vectors();
        }

        return cfg_.parse_terminate_;
    }

    scheduler_function partitioner::get_pool_creator(
        std::size_t index) const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (index >= initial_thread_pools_.size())
        {
            throw std::invalid_argument(
                "partitioner::get_pool_creator: pool requested out of bounds.");
        }
        return get_pool_data(index).create_function_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void partitioner::assign_pu(
        std::string const& pool_name, std::size_t virt_core)
    {
        std::lock_guard<mutex_type> l(mtx_);
        detail::init_pool_data& data = get_pool_data(pool_name);
        data.assign_pu(virt_core);
    }

    void partitioner::unassign_pu(
        std::string const& pool_name, std::size_t virt_core)
    {
        std::lock_guard<mutex_type> l(mtx_);
        detail::init_pool_data& data = get_pool_data(pool_name);
        data.unassign_pu(virt_core);
    }

    std::size_t partitioner::shrink_pool(std::string const& pool_name,
        util::function_nonser<void(std::size_t)> const& remove_pu)
    {
        if (get_runtime_ptr() == nullptr)
        {
            throw std::runtime_error("partitioner::create_thread_pool: "
                "this function must be called after the runtime system has "
                "been started");
        }

        if (!(mode_ & mode_allow_dynamic_pools))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "partitioner::shrink_pool",
                "dynamic pools have not been enabled for the "
                "partitioner");
        }

        std::vector<std::size_t> pu_nums_to_remove;
        bool has_non_exclusive_pus = false;

        {
            std::lock_guard<mutex_type> l(mtx_);
            detail::init_pool_data const& data = get_pool_data(pool_name);

            pu_nums_to_remove.reserve(data.num_threads_);

            for (std::size_t i = 0; i != data.num_threads_; ++i)
            {
                if (!data.pu_is_exclusive(i))
                {
                    has_non_exclusive_pus = true;
                    if (data.pu_is_assigned(i))
                    {
                        pu_nums_to_remove.push_back(i);
                    }
                }
            }
        }

        if (!has_non_exclusive_pus)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "partitioner::shrink_pool",
                "pool '" + pool_name + "' has no non-exclusive pus "
                "associated");
        }

        for (std::size_t pu_num : pu_nums_to_remove)
        {
            remove_pu(pu_num);
        }

        return pu_nums_to_remove.size();
    }

    std::size_t partitioner::expand_pool(std::string const& pool_name,
        util::function_nonser<void(std::size_t)> const& add_pu)
    {
        if (get_runtime_ptr() == nullptr)
        {
            throw std::runtime_error("partitioner::create_thread_pool: "
                "this function must be called after the runtime system has "
                "been started");
        }

        if (!(mode_ & mode_allow_dynamic_pools))
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "partitioner::expand_pool",
                "dynamic pools have not been enabled for the "
                "partitioner");
        }

        std::vector<std::size_t> pu_nums_to_add;
        bool has_non_exclusive_pus = false;

        {
            std::lock_guard<mutex_type> l(mtx_);
            detail::init_pool_data const& data = get_pool_data(pool_name);

            pu_nums_to_add.reserve(data.num_threads_);

            for (std::size_t i = 0; i != data.num_threads_; ++i)
            {
                if (!data.pu_is_exclusive(i))
                {
                    has_non_exclusive_pus = true;
                    if (!data.pu_is_assigned(i))
                    {
                        pu_nums_to_add.push_back(i);
                    }
                }
            }
        }

        if (!has_non_exclusive_pus)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "partitioner::expand_pool",
                "pool '" + pool_name + "' has no non-exclusive pus "
                "associated");
        }

        for (std::size_t pu_num : pu_nums_to_add)
        {
            add_pu(pu_num);
        }

        return pu_nums_to_add.size();
    }

    ////////////////////////////////////////////////////////////////////////
    std::size_t partitioner::get_pool_index(
        std::string const& pool_name) const
    {
        std::lock_guard<mutex_type> l(mtx_);

        std::size_t num_pools = initial_thread_pools_.size();
        for (std::size_t i = 0; i < num_pools; i++)
        {
            if (initial_thread_pools_[i].pool_name_ == pool_name)
            {
                return i;
            }
        }

        throw_invalid_argument(
            "partitioner::get_pool_index",
            "the resource partitioner does not own a thread pool named '" +
            pool_name + "'");
    }

    // has to be private bc pointers become invalid after data member
    // thread_pools_ is resized we don't want to allow the user to use it
    detail::init_pool_data const& partitioner::get_pool_data(
        std::string const& pool_name) const
    {
        auto pool = std::find_if(
            initial_thread_pools_.begin(), initial_thread_pools_.end(),
            [&pool_name](detail::init_pool_data const& itp) -> bool
            {
                return (itp.pool_name_ == pool_name);
            });

        if (pool != initial_thread_pools_.end())
        {
            return *pool;
        }

        throw_invalid_argument(
            "partitioner::get_pool_data",
            "the resource partitioner does not own a thread pool named '" +
            pool_name + "'");
    }

    detail::init_pool_data& partitioner::get_pool_data(
        std::string const& pool_name)
    {
        auto pool = std::find_if(
            initial_thread_pools_.begin(), initial_thread_pools_.end(),
            [&pool_name](detail::init_pool_data const& itp) -> bool
            {
                return (itp.pool_name_ == pool_name);
            });

        if (pool != initial_thread_pools_.end())
        {
            return *pool;
        }

        throw_invalid_argument(
            "partitioner::get_pool_data",
            "the resource partitioner does not own a thread pool named '" +
            pool_name + "'");
    }

    void partitioner::print_init_pool_data(std::ostream& os) const
    {
        std::lock_guard<mutex_type> l(mtx_);

        //! make this prettier
        os << "the resource partitioner owns "
            << initial_thread_pools_.size() << " pool(s) : \n";
        for (auto itp : initial_thread_pools_)
        {
            itp.print_pool(os);
        }
    }

    ////////////////////////////////////////////////////////////////////////
    boost::atomic<int> partitioner::instance_number_counter_(-1);
}}}
