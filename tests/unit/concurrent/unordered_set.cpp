//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/concurrent/unordered_set.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>
#include <iomanip>
#include <functional>
#include <string>
#include <vector>

#if HPX_HAVE_CXX14_LAMBDAS
# define QUOTED(x) std::quoted(x)
#else
# define QUOTED(x) x
#endif

// example code from cppreference.com
struct S {
    std::string first_name;
    std::string last_name;
};

bool operator==(const S& lhs, const S& rhs) {
    return lhs.first_name == rhs.first_name && lhs.last_name == rhs.last_name;
}

// custom hash can be a standalone function object:
template <class T> struct MyHash;
template<> struct MyHash<S>
{
    std::size_t operator()(S const& s) const
    {
        std::size_t h1 = std::hash<std::string>()(s.first_name);
        std::size_t h2 = std::hash<std::string>()(s.last_name);
        return h1 ^ (h2 << 1); // or use boost::hash_combine
    }
};

// custom specialization of std::hash can be injected in namespace std
namespace std
{
    template<> struct hash<S>
    {
        typedef S argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& s) const
        {
            result_type const h1 ( std::hash<std::string>()(s.first_name) );
            result_type const h2 ( std::hash<std::string>()(s.last_name) );
            return h1 ^ (h2 << 1); // or use boost::hash_combine
        }
    };
}

///////////////////////////////////////////////////////////////////////////////
void unordered_set_test1()
{
    std::string str = "Meet the new boss...";
    std::size_t str_hash = std::hash<std::string>{}(str);
    std::cout << "hash(" << QUOTED(str) << ") = " << str_hash << '\n';

    S obj = { "Hubert", "Farnsworth"};
    // using the standalone function object
    std::cout << "hash(" << QUOTED(obj.first_name) << ','
               << QUOTED(obj.last_name) << ") = "
               << MyHash<S>{}(obj) << " (using MyHash)\n                           or "
               << std::hash<S>{}(obj) << " (using std::hash) " << '\n';

    // custom hash makes it possible to use custom types in unordered containers
    hpx::concurrent::unordered_set<S> names = {obj, {"Bender", "Rodriguez"}, {"Leela", "Turanga"} };
    for(auto& s: names)
        std::cout << QUOTED(s.first_name) << ' ' << QUOTED(s.last_name) << '\n';

    //
    HPX_TEST(!names.empty());
}


///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    unordered_set_test1();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}


