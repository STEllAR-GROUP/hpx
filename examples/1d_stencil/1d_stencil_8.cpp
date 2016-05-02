//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is the eighth in a series of examples demonstrating the development
// of a fully distributed solver for a simple 1D heat distribution problem.
//
// This example builds on example seven.

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/lcos/gather.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <boost/shared_array.hpp>

#include "print_time_results.hpp"

#include <mutex>
#include <stack>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true; // print csv heading
double k = 0.5;     // heat transfer coefficient
double dt = 1.;     // time step
double dx = 1.;     // grid spacing

char const* stepper_basename = "/1d_stencil_8/stepper/";
char const* gather_basename = "/1d_stencil_8/gather/";

///////////////////////////////////////////////////////////////////////////////
// Use a special allocator for the partition data to remove a major contention
// point - the constant allocation and deallocation of the data arrays.
template <typename T>
struct partition_allocator
{
private:
    typedef hpx::lcos::local::spinlock mutex_type;

public:
    partition_allocator(std::size_t max_size = std::size_t(-1))
      : max_size_(max_size)
    {
    }

    ~partition_allocator()
    {
        std::lock_guard<mutex_type> l(mtx_);
        while (!heap_.empty())
        {
            T* p = heap_.top();
            heap_.pop();
            delete [] p;
        }
    }

    T* allocate(std::size_t n)
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (heap_.empty())
            return new T[n];

        T* next = heap_.top();
        heap_.pop();
        return next;
    }

    void deallocate(T* p)
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (max_size_ == static_cast<std::size_t>(-1) || heap_.size() < max_size_)
            heap_.push(p);
        else
            delete [] p;
    }

private:
    mutex_type mtx_;
    std::size_t max_size_;
    std::stack<T*> heap_;
};

///////////////////////////////////////////////////////////////////////////////
struct partition_data
{
private:
    typedef hpx::serialization::serialize_buffer<double> buffer_type;

    struct hold_reference
    {
        hold_reference(buffer_type const& data)
          : data_(data)
        {}

        void operator()(double*) {}     // no deletion necessary

        buffer_type data_;
    };

    static void deallocate(double* p)
    {
        alloc_.deallocate(p);
    }

    static partition_allocator<double> alloc_;

public:
    partition_data()
      : size_(0)
    {}

    // Create a new (uninitialized) partition of the given size.
    partition_data(std::size_t size)
      : data_(alloc_.allocate(size), size, buffer_type::take,
            &partition_data::deallocate),
        size_(size),
        min_index_(0)
    {}

    // Create a new (initialized) partition of the given size.
    partition_data(std::size_t size, double initial_value)
      : data_(alloc_.allocate(size), size, buffer_type::take,
            &partition_data::deallocate),
        size_(size),
        min_index_(0)
    {
        double base_value = double(initial_value * size);
        for (std::size_t i = 0; i != size; ++i)
            data_[i] = base_value + double(i);
    }

    // Create a partition which acts as a proxy to a part of the embedded array.
    // The proxy is assumed to refer to either the left or the right boundary
    // element.
    partition_data(partition_data const& base, std::size_t min_index)
      : data_(base.data_.data()+min_index, 1, buffer_type::reference,
            hold_reference(base.data_)),      // keep referenced partition alive
        size_(base.size()),
        min_index_(min_index)
    {
        HPX_ASSERT(min_index < base.size());
    }

    double& operator[](std::size_t idx) { return data_[index(idx)]; }
    double operator[](std::size_t idx) const { return data_[index(idx)]; }

    std::size_t size() const { return size_; }

private:
    std::size_t index(std::size_t idx) const
    {
        HPX_ASSERT(idx >= min_index_ && idx < size_);
        return idx - min_index_;
    }

private:
    // Serialization support: even if all of the code below runs on one
    // locality only, we need to provide an (empty) implementation for the
    // serialization as all arguments passed to actions have to support this.
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & data_ & size_ & min_index_;
    }

private:
    buffer_type data_;
    std::size_t size_;
    std::size_t min_index_;
};

partition_allocator<double> partition_data::alloc_;

std::ostream& operator<<(std::ostream& os, partition_data const& c)
{
    os << "{";
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << c[i];
    }
    os << "}";
    return os;
}

///////////////////////////////////////////////////////////////////////////////
inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if(i == 0 && dir == -1)
        return size-1;
    if(i == size-1 && dir == +1)
        return 0;

    HPX_ASSERT((i + dir) < size);

    return i + dir;
}

///////////////////////////////////////////////////////////////////////////////
// This is the server side representation of the data. We expose this as a HPX
// component which allows for it to be created and accessed remotely through
// a global address (hpx::id_type).
struct partition_server
  : hpx::components::component_base<partition_server>
{
    enum partition_type
    {
        left_partition, middle_partition, right_partition
    };

    // construct new instances
    partition_server() {}

    partition_server(partition_data const& data)
      : data_(data)
    {}

    partition_server(std::size_t size, double initial_value)
      : data_(size, initial_value)
    {}

    // Access data. The parameter specifies what part of the data should be
    // accessed. As long as the result is used locally, no data is copied,
    // however as soon as the result is requested from another locality only
    // the minimally required amount of data will go over the wire.
    partition_data get_data(partition_type t) const
    {
        switch (t)
        {
        case left_partition:
            return partition_data(data_, data_.size()-1);

        case middle_partition:
            break;

        case right_partition:
            return partition_data(data_, 0);

        default:
            HPX_ASSERT(false);
            break;
        }
        return data_;
    }

    // Every member function which has to be invoked remotely needs to be
    // wrapped into a component action. The macro below defines a new type
    // 'get_data_action' which represents the (possibly remote) member function
    // partition::get_data().
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(partition_server, get_data, get_data_action);

private:
    partition_data data_;
};

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<partition_server> partition_server_type;
HPX_REGISTER_COMPONENT(partition_server_type, partition_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef partition_server::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);

///////////////////////////////////////////////////////////////////////////////
// This is a client side helper class allowing to hide some of the tedious
// boilerplate while referencing a remote partition.
struct partition : hpx::components::client_base<partition, partition_server>
{
    typedef hpx::components::client_base<partition, partition_server> base_type;

    partition() {}

    // Create new component on locality 'where' and initialize the held data
    partition(hpx::id_type where, std::size_t size, double initial_value)
      : base_type(hpx::new_<partition_server>(where, size, initial_value))
    {}

    // Create a new component on the locality co-located to the id 'where'. The
    // new instance will be initialized from the given partition_data.
    partition(hpx::id_type where, partition_data const& data)
      : base_type(hpx::new_<partition_server>(hpx::colocated(where), data))
    {}

    // Attach a future representing a (possibly remote) partition.
    partition(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    partition(hpx::future<partition> && c)
      : base_type(std::move(c))
    {}

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<partition_data> get_data(partition_server::partition_type t) const
    {
        partition_server::get_data_action act;
        return hpx::async(act, get_id(), t);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Data for one time step on one locality
struct stepper_server : hpx::components::component_base<stepper_server>
{
    // Our data for one time step
    typedef std::vector<partition> space;

    stepper_server() {}

    stepper_server(std::size_t nl)
      : left_(hpx::find_from_basename(
            stepper_basename, idx(hpx::get_locality_id(), -1, nl))),
        right_(hpx::find_from_basename(
            stepper_basename, idx(hpx::get_locality_id(), +1, nl))),
        U_(2)
    {
    }

    // do all the work on 'np' local partitions, 'nx' data points each, for
    // 'nt' time steps
    space do_work(std::size_t local_np, std::size_t nx, std::size_t nt);

    HPX_DEFINE_COMPONENT_ACTION(stepper_server, do_work, do_work_action);

    // receive the left-most partition from the right
    void from_right(std::size_t t, partition p)
    {
        right_receive_buffer_.store_received(t, std::move(p));
    }

    // receive the right-most partition from the left
    void from_left(std::size_t t, partition p)
    {
        left_receive_buffer_.store_received(t, std::move(p));
    }

    HPX_DEFINE_COMPONENT_ACTION(stepper_server, from_right, from_right_action);
    HPX_DEFINE_COMPONENT_ACTION(stepper_server, from_left, from_left_action);

protected:
    // Our operator
    static double heat(double left, double middle, double right)
    {
        return middle + (k*dt/(dx*dx)) * (left - 2*middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition heat_part(partition const& left, partition const& middle,
        partition const& right);

    // Helper functions to receive the left and right boundary elements from
    // the neighbors.
    partition receive_left(std::size_t t)
    {
        return left_receive_buffer_.receive(t);
    }
    partition receive_right(std::size_t t)
    {
        return right_receive_buffer_.receive(t);
    }

    // Helper functions to send our left and right boundary elements to
    // the neighbors.
    void send_left(std::size_t t, partition p)
    {
        hpx::apply(from_right_action(), left_.get(), t, p);
    }
    void send_right(std::size_t t, partition p)
    {
        hpx::apply(from_left_action(), right_.get(), t, p);
    }

private:
    hpx::shared_future<hpx::id_type> left_, right_;
    std::vector<space> U_;
    hpx::lcos::local::receive_buffer<partition> left_receive_buffer_;
    hpx::lcos::local::receive_buffer<partition> right_receive_buffer_;
};

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation
// through hpx::new_<>().
typedef hpx::components::component<stepper_server> stepper_server_type;
HPX_REGISTER_COMPONENT(stepper_server_type, stepper_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef stepper_server::do_work_action do_work_action;
HPX_REGISTER_ACTION(do_work_action);

///////////////////////////////////////////////////////////////////////////////
// This is a client side member function can now be implemented as the
// stepper_server has been defined.
struct stepper : hpx::components::client_base<stepper, stepper_server>
{
    typedef hpx::components::client_base<stepper, stepper_server> base_type;

    // construct new instances/wrap existing steppers from other localities
    stepper()
      : base_type(hpx::new_<stepper_server>
          (hpx::find_here(), hpx::get_num_localities_sync()))
    {
        hpx::register_with_basename(
            stepper_basename, get_id(), hpx::get_locality_id());
    }

    stepper(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id))
    {}

    hpx::future<stepper_server::space> do_work(
        std::size_t local_np, std::size_t nx, std::size_t nt)
    {
        return hpx::async(do_work_action(), get_id(), local_np, nx, nt);
    }
};

///////////////////////////////////////////////////////////////////////////////
// The partitioned operator, it invokes the heat operator above on all elements
// of a partition.
partition stepper_server::heat_part(partition const& left,
    partition const& middle, partition const& right)
{
    using hpx::dataflow;
    using hpx::util::unwrapped;

    hpx::shared_future<partition_data> middle_data =
        middle.get_data(partition_server::middle_partition);

    hpx::future<partition_data> next_middle = middle_data.then(
        unwrapped(
            [middle](partition_data const& m) -> partition_data
            {
                // All local operations are performed once the middle data of
                // the previous time step becomes available.
                std::size_t size = m.size();
                partition_data next(size);
                for (std::size_t i = 1; i != size-1; ++i)
                    next[i] = heat(m[i-1], m[i], m[i+1]);
                return next;
            }
        )
    );

    return dataflow(
        hpx::launch::async,
        unwrapped(
            [left, middle, right](partition_data next, partition_data const& l,
                partition_data const& m, partition_data const& r) -> partition
            {
                // Calculate the missing boundary elements once the
                // corresponding data has become available.
                std::size_t size = m.size();
                next[0] = heat(l[size-1], m[0], m[1]);
                next[size-1] = heat(m[size-2], m[size-1], r[0]);

                // The new partition_data will be allocated on the same locality
                // as 'middle'.
                return partition(middle.get_id(), next);
            }
        ),
        std::move(next_middle),
        left.get_data(partition_server::left_partition),
        middle_data,
        right.get_data(partition_server::right_partition)
    );
}

///////////////////////////////////////////////////////////////////////////////
// This is the implementation of the time step loop
stepper_server::space stepper_server::do_work(std::size_t local_np,
    std::size_t nx, std::size_t nt)
{
    using hpx::dataflow;
    using hpx::util::unwrapped;

    // U[t][i] is the state of position i at time t.
    for (space& s: U_)
        s.resize(local_np);

    // Initial conditions: f(0, i) = i
    hpx::id_type here = hpx::find_here();
    for (std::size_t i = 0; i != local_np; ++i)
        U_[0][i] = partition(here, nx, double(i));

    // send initial values to neighbors
    send_left(0, U_[0][0]);
    send_right(0, U_[0][local_np-1]);

    for (std::size_t t = 0; t != nt; ++t)
    {
        space const& current = U_[t % 2];
        space& next = U_[(t + 1) % 2];

        // handle special case (one partition per locality) in a special way
        if (local_np == 1)
        {
            next[0] = dataflow(
                    hpx::launch::async, &stepper_server::heat_part,
                    receive_left(t), current[0], receive_right(t)
                );

            // send to left and right if not last time step
            if (t != nt-1)
            {
                send_left(t + 1, next[0]);
                send_right(t + 1, next[0]);
            }
        }
        else
        {
            next[0] = dataflow(
                    hpx::launch::async, &stepper_server::heat_part,
                    receive_left(t), current[0], current[1]
                );

            // send to left if not last time step
            if (t != nt-1) send_left(t + 1, next[0]);

            for (std::size_t i = 1; i != local_np-1; ++i)
            {
                next[i] = dataflow(
                        hpx::launch::async, &stepper_server::heat_part,
                        current[i-1], current[i], current[i+1]
                    );
            }

            next[local_np-1] = dataflow(
                    hpx::launch::async, &stepper_server::heat_part,
                    current[local_np-2], current[local_np-1], receive_right(t)
                );

            // send to right if not last time step
            if (t != nt-1) send_right(t + 1, next[local_np-1]);
        }
    }

    return U_[nt % 2];
}

HPX_REGISTER_GATHER(stepper_server::space, stepper_server_space_gatherer);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    using hpx::dataflow;

    boost::uint64_t nt = vm["nt"].as<boost::uint64_t>();   // Number of steps.
    boost::uint64_t nx = vm["nx"].as<boost::uint64_t>();   // Number of grid points.
    boost::uint64_t np = vm["np"].as<boost::uint64_t>();   // Number of partitions.

    if (vm.count("no-header"))
        header = false;

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();                    // Number of localities

    if (np < nl)
    {
        std::cout << "The number of partitions should not be smaller than "
                     "the number of localities" << std::endl;
        return hpx::finalize();
    }

    // Create the local stepper instance, register it
    stepper step;

    // Measure execution time.
    boost::uint64_t t = hpx::util::high_resolution_clock::now();

    // Perform all work and wait for it to finish
    hpx::future<stepper_server::space> result = step.do_work(np/nl, nx, nt);

    // Gather results from all localities
    if (0 == hpx::get_locality_id())
    {
        boost::uint64_t const num_worker_threads = hpx::get_num_worker_threads();

        hpx::future<std::vector<stepper_server::space> > overall_result =
            hpx::lcos::gather_here(gather_basename, std::move(result), nl);

        std::vector<stepper_server::space> solution = overall_result.get();
        for (std::size_t i = 0; i != nl; ++i)
        {
            stepper_server::space const& s = solution[i];
            for (std::size_t i = 0; i != s.size(); ++i)
                s[i].get_data(partition_server::middle_partition).wait();
        }

        boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

        // Print the solution at time-step 'nt'.
        if (vm.count("results"))
        {
            for (std::size_t i = 0; i != nl; ++i)
            {
                stepper_server::space const& s = solution[i];
                for (std::size_t j = 0; j != s.size(); ++j)
                {
                    std::cout << "U[" << i*(s.size()) + j << "] = "
                        << s[j].get_data(partition_server::middle_partition).get()
                        << std::endl;
                }
            }
        }

        print_time_results(boost::uint32_t(nl), num_worker_threads, elapsed,
            nx, np, nt, header);
    }
    else
    {
        hpx::lcos::gather_there(gather_basename, std::move(result)).wait();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("results", "print generated results (default: false)")
        ("nx", value<boost::uint64_t>()->default_value(10),
         "Local x dimension (of each partition)")
        ("nt", value<boost::uint64_t>()->default_value(45),
         "Number of time steps")
        ("np", value<boost::uint64_t>()->default_value(10),
         "Number of partitions")
        ("k", value<double>(&k)->default_value(0.5),
         "Heat transfer coefficient (default: 0.5)")
        ("dt", value<double>(&dt)->default_value(1.0),
         "Timestep unit (default: 1.0[s])")
        ("dx", value<double>(&dx)->default_value(1.0),
         "Local x dimension")
        ( "no-header", "do not print out the csv header row")
    ;

    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}
