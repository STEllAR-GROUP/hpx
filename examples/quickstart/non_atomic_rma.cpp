// Copyright (c) 2011 Matt Anderson <matt@phys.lsu.edu>
// Copyright (c) 2011 Pedro Diniz
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is the non-atomic version of the random memory access.  The array length
// is given by the variable array_length and N is the number of random accesses
// to this array.
//
//  At present, the entire array is placed in the root locality while all the random
//  accesses and writes occur on the remote locality.
//
// Note that this is a *non-atomic* example.

// HPX includes
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>

//Boost includes
#include <boost/program_options.hpp>

#include <boost/thread/locks.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <vector>

#include <iostream>
#include <ctime>

using namespace hpx;
namespace po = boost::program_options;

struct data
{
    data()
      : val_(0)
    {}
    ~data() {}

    int val_;
    std::vector<int> x_;
    bool proceed_;

private:
    // serialization support
    friend class hpx::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & val_ & x_ & proceed_;
    }
};

int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description desc_commandline ("usage:basic_example");

    int retcode = hpx::init(desc_commandline, argc, argv);
    return retcode;

}

naming::id_type set_initialdata(int);
void update(naming::id_type);

HPX_PLAIN_ACTION(set_initialdata);
HPX_PLAIN_ACTION(update);

int hpx_main(po::variables_map &vm)
{
    int result = 0;
    double elapsed = 0.0;

    std::vector<naming::id_type> localities =
        hpx::find_remote_localities();

    naming::id_type this_prefix = hpx::find_here();


    // Declaration used to store the first gid (if any) of the remote prefixes
    naming::id_type that_prefix;

    if (!localities.empty()) {
      // If there is at least one such remote locality, store the first gid in the list
      that_prefix = localities[0];
    } else {
      that_prefix = this_prefix;
    }


    {
        // Create a timer so see how its done
        util::high_resolution_timer t;

        std::vector<lcos::future<naming::id_type> > n;

        int array_length = 6;
        for (int i=0;i<array_length;i++) {
          n.push_back(hpx::async<set_initialdata_action>(this_prefix,i));
        }

        srand( time(NULL) );

        std::vector<lcos::future< void > > future_update;
        int N = 10; // number of random accesses to the array
        for (int i=0;i<N;i++) {
          int rn = rand() % array_length;
          std::cout << " Random number element accessed: " << rn << std::endl;
          naming::id_type tmp = n[rn].get();
          future_update.push_back(hpx::async<update_action>(that_prefix,tmp));
        }

        hpx::wait_all(future_update);

        for (int i=0;i<array_length;i++) {
          components::access_memory_block<data>
                  result( components::stubs::memory_block::get_data(n[i].get()) );
          std::cout << " Result index: " << i << " value : "
                  << result->val_ << std::endl;
        }

        // What is the elapsed time?
        elapsed = t.elapsed();

        // Print out a completion message. The correct answer is 12 for this example.
        std::cout << "Achieved result of " << result << " in "
            << elapsed << " seconds."<< std::endl;
    }

    // Initiate shutdown of the runtime systems on all localities
    hpx::finalize(5.0);
    return 0;
}

hpx::actions::manage_object_action<data> const manage_data =
        hpx::actions::manage_object_action<data>();

naming::id_type set_initialdata (int i)
{

  naming::id_type here = hpx::find_here();
    naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(data), manage_data);

    components::access_memory_block<data> val(
                components::stubs::memory_block::checkout(result));

    int locality = get_locality_id_from_id( here );

    val->val_ = i;
    std::cout << " locality : " << locality << " index : " << i << std::endl;

    return result;
}

// the "work"
void update (naming::id_type in)
{

    components::access_memory_block<data> result(
                components::stubs::memory_block::checkout(in));

    naming::id_type here = applier::get_applier().get_runtime_support_gid();
    int locality = get_locality_id_from_id( here );
    std::cout << " locality update " << locality << std::endl;

    result->val_ += 1;

    components::stubs::memory_block::checkin(in, result);
}

