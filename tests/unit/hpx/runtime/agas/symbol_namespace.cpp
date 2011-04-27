//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/unordered_set.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/agas/namespace/symbol.hpp>
#include <hpx/runtime/agas/database/backend/default.hpp>
#include <hpx/lcos/local_barrier.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::mt19937;
using boost::uniform_int;

using boost::ref;
using boost::cref;

using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::agas::symbol_namespace;

using hpx::lcos::local_barrier;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef hpx::agas::symbol_namespace<
    hpx::agas::tag::database::default_
> symbol_namespace_type;


void insert_key_value_pair (local_barrier& global, local_barrier& barr,
                            symbol_namespace_type& sym, std::string const& key,
                            gid_type const& value)
{
    // Insert the key and value synchronously. 
    bool r = sym.bind(key, value);

    // Check if the insertion succeeded.
    HPX_TEST(r);

    // Now that we've inserted the key, wait on the barrier for this entry. The
    // thread that will be resolving this key waits on this barrier before it
    // attempts the resolve action (this makes the test thread-safe).
    barr.wait();

    // Suspend until shutdown.
    global.wait();
}

void resolve_key_value_pair (local_barrier& global, local_barrier& barr,
                             symbol_namespace_type& sym, std::string const& key,
                             gid_type const& value)
{
    // Wait until the insertion has been completed.
    barr.wait();

    // Resolve the key synchronously. 
    gid_type r = sym.resolve(key);

    // Verify the resolved GID's value. 
    HPX_TEST_EQ(r, value);

    // Suspend until shutdown.
    global.wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t entries = 0;

    if (vm.count("entries"))
        entries = vm["entries"].as<std::size_t>();

    boost::unordered_set<std::string> keys;

    // Alphabet for keys.
    std::string const chars(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "1234567890");        

    mt19937 rng;

    // Random number distributions.
    uniform_int<> index_dist(0, chars.size() - 1);
    uniform_int<std::size_t> length_dist(4, 18);
    uniform_int<boost::uint64_t> gid_dist;

    // Get this locality's prefix.
    id_type prefix = get_applier().get_runtime_support_gid();

    // Create the symbol namespace.
    symbol_namespace_type sym;
    sym.create(prefix);
        
    // Create the global barrier for shutdown.
    local_barrier global(entries + 1);

    // Allocate the storage for the local barriers
    boost::scoped_array<local_barrier> barriers
        (reinterpret_cast<local_barrier*>(::operator new
            (sizeof(local_barrier) * entries)));

    for (std::size_t i = 0; i < entries; ++i) {
        std::size_t const len = length_dist(rng);

        // Construct the barrier for this entry
        new (&barriers[i]) local_barrier(2);

        // Have the key preallocate space to avoid multiple resizes.
        std::string key;

        // Generate a unique key.
        do {
            key.clear();
            key.reserve(len);
            for (std::size_t j = 0; j < len; ++j)
                key.push_back(chars[index_dist(rng)]);
        } while (keys.count(key));
 
        gid_type value(gid_dist(rng), gid_dist(rng));

        register_work(boost::bind(&insert_key_value_pair,
            ref(global), ref(barriers[i]), ref(sym), cref(key), cref(value)));
        
        register_work(boost::bind(&resolve_key_value_pair,
            ref(global), ref(barriers[i]), ref(sym), cref(key), cref(value)));
    }

    // wait for all threads to enter the barrier
    global.wait(); 

    // initiate shutdown of the runtime system
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("entries", value<std::size_t>()->default_value(1 << 6), 
            "the number of entries used in each iteration") 
        //("iterations", value<std::size_t>()->default_value(1 << 6), 
        //    "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

