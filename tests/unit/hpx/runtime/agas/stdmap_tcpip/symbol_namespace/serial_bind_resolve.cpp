//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/integer.hpp>
#include <boost/unordered_set.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/agas/namespace/user_symbol.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::mt19937;
using boost::uniform_int;

using boost::integer_traits;

using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::applier::get_applier;
using hpx::applier::register_work;

using hpx::agas::user_symbol_namespace;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef user_symbol_namespace<
    hpx::agas::tag::database::stdmap
> symbol_namespace_type;

void insert_key_value_pair (symbol_namespace_type& sym, std::string const& key,
                            gid_type const& value)
{
    // Insert the key and value synchronously. 
    bool r = sym.bind(key, value);

    // Check if the insertion succeeded.
    HPX_TEST(r);
}

void resolve_key_value_pair (symbol_namespace_type& sym, std::string const& key,
                             gid_type const& value)
{
    // Resolve the key synchronously. 
    gid_type r = sym.resolve(key);

    // Verify the resolved GID's value. 
    HPX_TEST_EQ(r, value);
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
    {
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
        uniform_int<boost::uint64_t> gid_dist(1,
            integer_traits<boost::uint64_t>::const_max);
    
        // Get this locality's prefix.
        id_type prefix = get_applier().get_runtime_support_gid();
    
        // Create the symbol namespace.
        symbol_namespace_type sym;
        sym.create(prefix);
            
        for (std::size_t e = 0; e < entries; ++e)
        {
            std::size_t const len = length_dist(rng);
    
            // Have the key preallocate space to avoid multiple resizes.
            std::string key;
    
            // Generate a unique key.
            do {
                key.clear();
                key.reserve(len);
                for (std::size_t j = 0; j < len; ++j)
                    key.push_back(chars[index_dist(rng)]);
            } while (keys.count(key));
   
            keys.insert(key);
  
            gid_type value(gid_dist(rng), gid_dist(rng));
    
            insert_key_value_pair(sym, key, value);
           
            resolve_key_value_pair(sym, key, value); 
        }
    }

    // initiate shutdown of the runtime system
    finalize();
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

