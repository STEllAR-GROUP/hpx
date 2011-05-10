//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>

#include <boost/atomic.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/future_wait.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::components::memory_block;
using hpx::components::access_memory_block;

using hpx::actions::manage_object_action;
using hpx::actions::plain_action4;

using hpx::applier::applier;
using hpx::applier::get_applier;

using hpx::lcos::eager_future;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::size_t partition(T* data, std::size_t begin, std::size_t end)
{
    T* first = data + begin;
    T* last = first + (end - begin);

    T* middle = std::partition(
        first, last, std::bind2nd(std::less<T>(), *first));

    return middle - data;
}
 
///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct quicksort
{
    static std::size_t sort_count;

    static void call(id_type prefix, id_type d, std::size_t begin,
                     std::size_t end);

    typedef plain_action4<
        id_type, id_type, std::size_t, std::size_t, &quicksort::call
    > action_type;
};

template <typename T>
std::size_t quicksort<T>::sort_count(0); 

template <typename T>
void quicksort<T>::call(id_type prefix, id_type d, std::size_t begin,
                        std::size_t end)
{
    if (begin != end) {
        memory_block mb(d);
        access_memory_block<T> data(mb.get());

        std::size_t middle_idx = partition(data.get_ptr(), begin, end);

        ++sort_count;

        // always spawn the larger part in a new thread
        if (2 * middle_idx < end - begin) {
            eager_future<action_type> n(prefix, prefix, d, 
                (std::max)(begin + 1, middle_idx), end);

            call(prefix, d, begin, middle_idx);
            ::hpx::components::wait(n);
        }

        else {
            eager_future<action_type> n(prefix, prefix, d, 
                begin, middle_idx);

            call(prefix, d, (std::max)(begin + 1, middle_idx), end);
            ::hpx::components::wait(n);
        }
    }
}

typedef quicksort<int>::action_type quicksort_int_action;
HPX_REGISTER_PLAIN_ACTION(quicksort_int_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t elements = 0;

    if (vm.count("elements"))
        elements = vm["elements"].as<std::size_t>();

    manage_object_action<boost::uint8_t> const raw_memory =
      manage_object_action<boost::uint8_t>();

    // get list of all known localities
    std::vector<id_type> prefixes;
    id_type prefix;

    applier& appl = get_applier();

    // execute the qsort() function on any of the remote localities
    if (appl.get_remote_prefixes(prefixes))
        prefix = prefixes[0];

    // execute the qsort() function locally
    else
        prefix = appl.get_runtime_support_gid();

    {
        // create a (remote) memory block
        memory_block mb;
        mb.create(prefix, sizeof(int) * elements, raw_memory);
        access_memory_block<int> data(mb.get());

        // randomly fill the vector
        std::generate(data.get_ptr(), data.get_ptr() + elements, std::rand);

        high_resolution_timer t;
        std::sort(data.get_ptr(), data.get_ptr() + elements);

        double elapsed = t.elapsed();
        std::cout << "elapsed: " << elapsed << std::endl;

        std::generate(data.get_ptr(), data.get_ptr() + elements, std::rand);
        t.restart();

        eager_future<quicksort<int>::action_type> n(
            prefix, prefix, mb.get_gid(), 0, elements);
        ::hpx::components::wait(n);

        elapsed = t.elapsed();
        std::cout << "elapsed: " << elapsed << std::endl;
        std::cout << "count: " << quicksort<int>::sort_count << std::endl;

        mb.free();
    }

    // initiate shutdown of the runtime systems on all localities
    finalize(5.0);

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("elements", value<std::size_t>()->default_value(1024), 
            "the number of elements to generate and sort") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

