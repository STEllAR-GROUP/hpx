//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates the use of the hpx::lcos::local::receive_buffer
// facility It can be used to decouple time-step based operations between
// remote partitions of a spatially decomposed problem.

// Including 'hpx/hpx_main.hpp' instead of the usual 'hpx/hpx_init.hpp' enables
// to use the plain C-main below as the direct main HPX entry point.
#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>

#include <cstddef>
#include <deque>
#include <utility>

// This example assumes that the computational space is divided into two
// partitions. Here we place both partitions on the same locality, but they
// could be easily placed on different HPX localities without any changes to
// the code as well.
//
// The example rotates the data stored in both partitions by one position
// during each time step while making sure that the right-most value is
// transferred to the neighboring partition.
//
// Each partition is represented by a component type 'partition' which holds
// the data (here a simple std::deque<int>) and an instance of a
// receive_buffer for the neighboring partition. The two partitions in
// this example are connected in a circular fashion, thus both of them have
// one receive_buffer, always representing the data to be received from the
// 'left'.

char const* partition_basename = "/receive_buffer_example/partition/";

// The neighbor of partition '0' is partition '1', and v.v.
std::size_t neighbor(std::size_t partition_num)
{
    return partition_num == 0 ? 1 : 0;
}

///////////////////////////////////////////////////////////////////////////////
struct partition_server : hpx::components::component_base<partition_server>
{
    partition_server() {}

    // Retrieve the neighboring partition
    partition_server(std::size_t partition_num, std::size_t num_elements)
      : data_(num_elements),
        left_(hpx::find_from_basename(partition_basename, neighbor(partition_num)))
    {
        // fill with some random data
        std::generate(data_.begin(), data_.end(), std::rand);
    }

public:
    // Action definitions

    // Do all the work for 'nt' time steps on the local
    hpx::future<void> do_work(std::size_t nt);
    HPX_DEFINE_COMPONENT_ACTION(partition_server, do_work, do_work_action);

    // Receive the data from the left partition. This will be called by the
    // other partition, sending us its data.
    void from_right(std::size_t timestep, int data)
    {
        right_buffer_.store_received(timestep, std::move(data));
    }
    HPX_DEFINE_COMPONENT_ACTION(partition_server, from_right, from_right_action);

    // Explicitly release dependencies to avoid circular dependencies in the
    // reference counting chain.
    void release_dependencies()
    {
        left_.free();
    }
    HPX_DEFINE_COMPONENT_ACTION(partition_server, release_dependencies,
        release_dependencies_action);

public:
    // Other helper functions

    // Helper function to send our boundary elements to the left neighbor.
    void send_left(std::size_t timestep, int data) const
    {
        hpx::apply(from_right_action(), left_, timestep, data);
    }

    // Helper function to receive the boundary element from the right neighbor.
    hpx::future<int> receive_right(std::size_t timestep)
    {
        return right_buffer_.receive(timestep);
    }

private:
    // Data stored in this partition.
    std::deque<int> data_;

    // The id held by the future represents the neighboring partition (the one
    // where the next element should be sent to).
    hpx::components::client<partition_server> left_;

    // The receive buffers represents one single int to be received from the
    // corresponding neighbor.
    hpx::lcos::local::receive_buffer<int> right_buffer_;
};

// The macros below are necessary to generate the code required for exposing
// our partition type remotely.
//
// HPX_REGISTER_COMPONENT() exposes the component creation through hpx::new_<>().
typedef hpx::components::component<partition_server> partition_server_type;
HPX_REGISTER_COMPONENT(partition_server_type, partition_server);

// HPX_REGISTER_ACTION() exposes the component member function for remote
// invocation.
typedef partition_server::from_right_action from_right_action;
HPX_REGISTER_ACTION(from_right_action);

typedef partition_server::do_work_action do_work_action;
HPX_REGISTER_ACTION(do_work_action);

typedef partition_server::release_dependencies_action release_dependencies_action;
HPX_REGISTER_ACTION(release_dependencies_action);

///////////////////////////////////////////////////////////////////////////////
struct partition : hpx::components::client_base<partition, partition_server>
{
    typedef hpx::components::client_base<partition, partition_server> base_type;

    partition(hpx::id_type const& locality, std::size_t partition_num,
            std::size_t num_elements)
      : base_type(hpx::new_<partition_server>(
                locality, partition_num, num_elements
            )),
        partition_num_(partition_num),
        registered_name_(true)
    {
        // Register this partition with the runtime so that its neighbor can
        // find it.
        hpx::register_with_basename(partition_basename, *this, partition_num).get();
    }

    partition(hpx::future<hpx::id_type> && id)
      : base_type(std::move(id)),
        registered_name_(false)
    {}

    ~partition()
    {
        if (!registered_name_)
            return;

        // break cyclic dependencies
        hpx::future<void> f1 = hpx::async(release_dependencies_action(), *this);

        // release the reference held by AGAS
        hpx::future<void> f2 = hpx::unregister_with_basename(
            partition_basename, partition_num_);

        hpx::wait_all(f1, f2);      // ignore exceptions
    }

    hpx::future<void> do_work(std::size_t nt)
    {
        return hpx::async(do_work_action(), *this, nt);
    }

private:
    std::size_t partition_num_;
    bool registered_name_;
};

///////////////////////////////////////////////////////////////////////////////
// This is the implementation of the time step loop
hpx::future<void> partition_server::do_work(std::size_t nt)
{
    // send initial values to neighbors
    if (nt != 0)
    {
        // send left-most element
        send_left(0, data_[0]);

        // rotate left by one element
        std::rotate(data_.begin(), data_.begin() + 1, data_.end());
    }

    hpx::future<void> result = hpx::make_ready_future();
    for (std::size_t t = 0; t != nt; ++t)
    {
        // Receive element from the right, replace last local element with the
        // received value.
        //
        // Each timestep depends on a) the previous timestep and b) the
        // received value for the current timestep.
        result =
            hpx::dataflow(
                [this, t, nt](hpx::future<void> result, hpx::future<int> f)
                {
                    result.get();       // propagate exceptions

                    // replace right-most element with received value
                    data_[data_.size()-1] = f.get();

                    // if not last time step, send left-most and rotate left
                    // by one element
                    if (t != nt - 1)
                    {
                        send_left(t + 1, data_[0]);
                        std::rotate(data_.begin(), data_.begin() + 1, data_.end());
                    }
                },
                result, receive_right(t));
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initial conditions: f(0, i) = i
    hpx::id_type here = hpx::find_here();

    // create partitions and launch work
    partition p0(here, 0, 1000);
    hpx::future<void> f0 = p0.do_work(100);

    partition p1(here, 1, 1000);
    hpx::future<void> f1 = p1.do_work(100);

    // wait for both partitions to be finished
    hpx::wait_all(f0, f1);

    return 0;
}

