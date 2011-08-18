//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/unordered_map.hpp>
#include <boost/integer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/agas/namespace/user_symbol.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/lcos/local_barrier.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::mt19937;
using boost::uniform_int;

using boost::integer_traits;

using boost::ref;
using boost::cref;

using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::agas::user_symbol_namespace;

using hpx::lcos::local_barrier;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef boost::unordered_map<std::string, gid_type> entry_table_type;

template <typename Database>
void bind_symbol (local_barrier& barr, symbol_namespace<Database>& sym,
                  std::string const& key, gid_type const& value)
{
    // Insert the key and value synchronously. 
    bool r = sym.bind(key, value);

    // Check if the insertion succeeded.
    HPX_TEST(r);

    // Now that we've inserted the key, wait on the barrier for this entry. The
    // thread that will be resolving this key waits on this barrier before it
    // attempts the resolve action (this makes the test thread-safe).
    barr.wait();
}

template <typename Database>
void resolve_symbol (local_barrier& global, local_barrier& barr,
                     symbol_namespace<Database>& sym, std::string const& key,
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

template <typename Database>
void test_symbol_namespace(std::size_t entries)
{
    entry_table_type entry_table;

    // Alphabet for keys.
    std::string const chars(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "1234567890");        

    mt19937 rng;

    // Random number distributions.
    uniform_int<> index_dist(0, chars.size() - 1);
    uniform_int<std::size_t> length_dist(4, 18);
    uniform_int<boost::uint64_t> gid_dist(1,
        integer_traits<boost::uint64_t>::const_max);

    // Get this locality's prefix.
    id_type prefix = get_applier().get_runtime_support_gid();

    // Create the symbol namespace.
    user_symbol_namespace<Database> sym;
    sym.create(prefix);
        
    // Create the global barrier for shutdown.
    local_barrier global(entries + 1);

    // Allocate the storage for the local barriers
    local_barrier* barriers
        (reinterpret_cast<local_barrier*>(::malloc
            (sizeof(local_barrier) * entries)));

    for (std::size_t e = 0; e < entries; ++e)
    {
        std::size_t const len = length_dist(rng);

        // Construct the barrier for this entry
        new (&barriers[e]) local_barrier(2);

        // Have the key preallocate space to avoid multiple resizes.
        std::string key;

        // Generate a unique key.
        do {
            key.clear();
            key.reserve(len);
            for (std::size_t j = 0; j < len; ++j)
                key.push_back(chars[index_dist(rng)]);
        } while (entry_table.count(key));

        gid_type value(gid_dist(rng), gid_dist(rng));

        entry_table_type::iterator it = entry_table.insert
            (entry_table_type::value_type(key, value)).first;

        register_work(boost::bind(&bind_symbol<Database>,
            ref(barriers[e]), ref(sym), cref(it->first), cref(it->second)));
        
        register_work(boost::bind(&resolve_symbol<Database>,
            ref(global), ref(barriers[e]), ref(sym), cref(it->first),
            cref(it->second)));
    }

    // wait for all threads to enter the barrier
    global.wait(); 

    ::free(reinterpret_cast<void*>(barriers));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t entries = 0;

    if (vm.count("entries"))
        entries = vm["entries"].as<std::size_t>();

    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();

    for (std::size_t i = 0; i < iterations; ++i)
        test_symbol_namespace<hpx::agas::tag::database::stdmap>(entries);

    // initiate shutdown of the runtime system
    finalize(5.0);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("entries", value<std::size_t>()->default_value(1 << 6), 
            "the number of entries used in each iteration") 
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

