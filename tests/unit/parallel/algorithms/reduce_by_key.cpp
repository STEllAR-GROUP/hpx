//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/parallel/algorithms/sort_by_key.hpp>
#include <hpx/parallel/algorithms/reduce_by_key.hpp>
//
#include <boost/random/uniform_int_distribution.hpp>
//
#define HPX_REDUCE_BY_KEY_TEST_SIZE (1 << 4)
//
#include "sort_tests.hpp"
//
#define EXTRA_DEBUG
//
namespace debug {
    template<typename T>
    void output(const std::string &name, const std::vector<T> &v) {
#ifdef EXTRA_DEBUG
        std::cout << name.c_str() << "\t : {" << v.size() << "} : ";
        std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(std::cout, ", "));
        std::cout << "\n";
#endif
    }

    template<typename Iter>
    void output(const std::string &name, Iter begin, Iter end) {
#ifdef EXTRA_DEBUG
        std::cout << name.c_str() << "\t : {" << std::distance(begin,end) << "} : ";
        std::copy(begin, end,
                  std::ostream_iterator<typename std::iterator_traits<Iter>::value_type>(std::cout, ", "));
        std::cout << "\n";
#endif
    }
#if defined(EXTRA_DEBUG)
# define debug_msg(a) std::cout << a
#else
# define debug_msg(a)
#endif
};

#undef msg
#define msg(a,b,c,d,e) \
        std::cout \
        << std::setw(60) << a << std::setw(12) <<  b \
        << std::setw(40) << c << std::setw(6)  <<  d \
        << std::setw(8)  << #e << "\t";

////////////////////////////////////////////////////////////////////////////////
// call reduce_by_key with no comparison operator
template <typename ExPolicy, typename Tkey, typename Tval, typename Op, typename HelperOp>
void test_reduce_by_key1(ExPolicy && policy, Tkey, Tval, const Op &op, const HelperOp &ho)
{
    static_assert(
            hpx::parallel::is_execution_policy<ExPolicy>::value,
            "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(Tval).name(), typeid(Op).name(), "default", sync);
    std::cout << "\n";

    Tval rnd_min = -100;
    Tval rnd_max =  100;

    // vector of values, and keys
    std::vector<Tval> values;
    std::vector<Tkey> keys;
    values.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);
    keys.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);

    // to simply validate our reduction we will use a map
    std::map<Tkey,Tval> key_check;

    // use the default random engine and an uniform distribution for values
    boost::random::mt19937 eng(static_cast<unsigned int>(std::rand()));
    boost::random::uniform_real_distribution<double> distr(rnd_min, rnd_max);

    // use the default random engine and an uniform distribution for keys
    boost::random::mt19937 engk(static_cast<unsigned int>(std::rand()));
    boost::random::uniform_real_distribution<double> distrk(0, 256);
    // generate test data
    for (int i=0; i<HPX_REDUCE_BY_KEY_TEST_SIZE; i++) {
        Tkey key = static_cast<Tkey>(distrk(engk));
        Tval value = static_cast<Tval>(distr(eng));
        Tkey helperkey = ho(key);
        if (key_check.find(helperkey)==key_check.end()) {
            key_check[helperkey] = value;
        }
        else {
            Tval temp = key_check[helperkey] + value;
            key_check[helperkey] = temp;
        }
        keys.push_back(key);
        values.push_back(value);
    }

    hpx::parallel::sort_by_key(
            hpx::parallel::seq,
            keys.begin(),
            keys.end(),
            values.begin());

    // output
    //debug::output("\nkeys", keys);
    //debug::output("\nvalues", values);

    std::vector<Tval> check_values;
    for (typename std::map<Tkey,Tval>::iterator it=key_check.begin(); it!=key_check.end(); ++it) {
        check_values.push_back(it->second);
        //debug_msg("Operation of " << it->first << " is " << it->second << "\n");
    }

    auto policy2 = hpx::parallel::par.with(hpx::parallel::static_chunk_size(4096));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // reduce_by_key, blocking when seq, par, par_vec
    auto result = hpx::parallel::reduce_by_key(
            policy2,
            //std::forward<ExPolicy>(policy),
            keys.begin(), keys.end(),
            values.begin(),
            keys.begin(),
            values.begin(),
            op);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_equal = std::equal(values.begin(), result.second, check_values.begin());
    if (is_equal) {
        std::cout << "Test Passed\n";
    }
    else {
        debug::output("key range", keys.begin(), result.first);
        debug::output("val range", values.begin(), result.second);
        debug::output("expected ", check_values);
    }
    HPX_TEST(is_equal);
}

////////////////////////////////////////////////////////////////////////////////
void test_reduce_by_key1()
{
    using namespace hpx::parallel;

    // default comparison operator (std::equal_to)
    test_reduce_by_key1(seq,     int(), int(), std::equal_to<int>(), [](int a){return a;});
    test_reduce_by_key1(par,     int(), int(), std::equal_to<int>(), [](int a){return a;});
    test_reduce_by_key1(par_vec, int(), int(), std::equal_to<int>(), [](int a){return a;});

    // default comparison operator (std::equal_to)
    test_reduce_by_key1(seq,     int(), double(), std::equal_to<double>(), [](int a){return a;});
    test_reduce_by_key1(par,     int(), double(), std::equal_to<double>(), [](int a){return a;});
    test_reduce_by_key1(par_vec, int(), double(), std::equal_to<double>(), [](int a){return a;});

    //
    test_reduce_by_key1(seq,     double(), double(),
                        [](double a, double b) { return std::floor(a)==std::floor(b); },
                        [](double a){ return std::floor(a); }
    );
    test_reduce_by_key1(par,     double(), double(),
                        [](double a, double b) { return std::floor(a)==std::floor(b); },
                        [](double a){ return std::floor(a); }
    );
    test_reduce_by_key1(par_vec,     double(), double(),
                        [](double a, double b) { return std::floor(a)==std::floor(b); },
                        [](double a){ return std::floor(a); }
    );

/*
    test_reduce_by_key1(seq,     double(), std::multiplies<double>());
    test_reduce_by_key1(par,     double(), std::multiplies<double>());
    test_reduce_by_key1(par_vec, double(), std::multiplies<double>());

    // user supplied comparison operator (std::less)
    test_reduce_by_key1_comp(seq,     int(), std::less<std::size_t>());
    test_reduce_by_key1_comp(par,     int(), std::less<std::size_t>());
    test_reduce_by_key1_comp(par_vec, int(), std::less<std::size_t>());

    // user supplied comparison operator (std::greater)
    test_reduce_by_key1_comp(seq,     double(), std::greater<double>());
    test_reduce_by_key1_comp(par,     double(), std::greater<double>());
    test_reduce_by_key1_comp(par_vec, double(), std::greater<double>());

    // Async execution, default comparison operator
    test_reduce_by_key1_async(seq(task), int());
    test_reduce_by_key1_async(par(task), char());
    test_reduce_by_key1_async(seq(task), double());
    test_reduce_by_key1_async(par(task), float());
    test_reduce_by_key1_async_str(seq(task));
    test_reduce_by_key1_async_str(par(task));

    // Async execution, user comparison operator
    test_reduce_by_key1_async(seq(task), int(),    std::less<unsigned int>());
    test_reduce_by_key1_async(par(task), char(),   std::less<char>());
    //
    test_reduce_by_key1_async(seq(task), double(), std::greater<double>());
    test_reduce_by_key1_async(par(task), float(),  std::greater<float>());
    //
    test_reduce_by_key1_async_str(seq(task), std::greater<std::string>());
    test_reduce_by_key1_async_str(par(task), std::greater<std::string>());

    test_reduce_by_key1(execution_policy(seq),       int());
    test_reduce_by_key1(execution_policy(par),       int());
    test_reduce_by_key1(execution_policy(par_vec),   int());
    test_reduce_by_key1(execution_policy(seq(task)), int());
    test_reduce_by_key1(execution_policy(par(task)), int());
    test_reduce_by_key1(execution_policy(seq(task)), std::string());
    test_reduce_by_key1(execution_policy(par(task)), std::string());
*/
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_reduce_by_key1();
//    test_reduce_by_key2();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
