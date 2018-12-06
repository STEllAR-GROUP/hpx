//  Copyright (c) 2015-2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <hpx/parallel/algorithms/reduce_by_key.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <random>
#include <utility>
#include <vector>
#ifdef EXTRA_DEBUG
# include <cstddef>
# include <string>
# include <iostream>
#endif
//
#define HPX_REDUCE_BY_KEY_TEST_SIZE (1 << 18)
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
            std::ostream_iterator<typename std::iterator_traits<Iter>::
            value_type>(std::cout, ", "));
        std::cout << "\n";
#endif
    }
#if defined(EXTRA_DEBUG)
# define debug_msg(a) std::cout << a
#else
# define debug_msg(a)
#endif
}
;

#undef msg
#define msg(a,b,c,d) \
        std::cout \
        << std::setw(60) << a << std::setw(12) <<  b \
        << std::setw(40) << c << std::setw(30) \
        << std::setw(8)  << #d << "\t";

struct almost_equal
{
    template <typename T>
    bool operator()(T lhs, T rhs) const
    {
        return lhs == rhs;
    }
    bool operator()(float lhs, float rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10f;
    }
    bool operator()(double lhs, double rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10;
    }
    template <typename T>
    bool operator()(T lhs, double rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10;
    }
    template <typename T>
    bool operator()(double lhs, T rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10;
    }
    template <typename T>
    bool operator()(T lhs, float rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10f;
    }
    template <typename T>
    bool operator()(float lhs, T rhs) const
    {
        return std::abs(lhs - rhs) < 1e-10;
    }
};


////////////////////////////////////////////////////////////////////////////////
int seed = std::random_device{}();
std::mt19937 gen(seed);

template<typename ExPolicy, typename Tkey, typename Tval, typename Op, typename HelperOp>
void test_reduce_by_key1(ExPolicy && policy, Tkey, Tval, bool benchmark, const Op &op,
    const HelperOp &ho)
    {
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(Tval).name(), typeid(Op).name(), sync);
    std::cout << "\n";

    Tval rnd_min = -256;
    Tval rnd_max = 256;

    // vector of values, and keys
    std::vector<Tval> values, o_values;
    std::vector<Tkey> keys, o_keys;
    values.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);
    keys.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);

    std::vector<Tval> check_values;

    // use the default random engine and an uniform distribution for values
    std::mt19937 eng(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distr(rnd_min, rnd_max);

    // use the default random engine and an uniform distribution for keys
    std::mt19937 engk(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distrk(0, 256);

    // generate test data
    int keysize = 0;
    Tkey key = 0, helperkey = 0, lastkey = 0;
    for (/* */; keysize < HPX_REDUCE_BY_KEY_TEST_SIZE;)
        {
        do {
            key = static_cast<Tkey>(distrk(engk));
        } while (ho(key) == lastkey);
        helperkey = ho(key);
        lastkey = helperkey;
        //
        int numkeys = static_cast<int>(distrk(engk)) + 1;
        //
        Tval sum = 0;
        for (int i = 0; i < numkeys && keysize < HPX_REDUCE_BY_KEY_TEST_SIZE; ++i) {
            Tval value = static_cast<Tval>(distr(eng));
            keys.push_back(key);
            values.push_back(value);
            sum += value;
            keysize++;
        }
        check_values.push_back(sum);
    }
    o_values = values;
    o_keys = keys;

    hpx::util::high_resolution_timer t;
    // reduce_by_key, blocking when seq, par, par_vec
    auto result = hpx::parallel::reduce_by_key(
        std::forward<ExPolicy>(policy),
        keys.begin(), keys.end(),
        values.begin(),
        keys.begin(),
        values.begin(),
        op);
    double elapsed = t.elapsed();

    bool is_equal = std::equal(values.begin(), result.second, check_values.begin(),
        almost_equal());
    HPX_TEST(is_equal);
    if (is_equal) {
        if (benchmark) {
            // CDash graph plotting
            hpx::util::print_cdash_timing("ReduceByKeyTime", elapsed);
        }
    }
    else {
//         debug::output("keys     ", o_keys);
//         debug::output("values   ", o_values);
//         debug::output("key range", keys.begin(), result.first);
//         debug::output("val range", values.begin(), result.second);
//         debug::output("expected ", check_values);
//         throw std::string("Problem");
#if defined(EXTRA_DEBUG)
        for (std::size_t i = 0; i != check_values.size(); ++i)
        {
            if (!almost_equal()(values[i],check_values[i]))
            {
                std::cout
                    << i << ": "
                    << values[i] << " != " << check_values[i]
                    << "\n";
            }
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
template<typename ExPolicy, typename Tkey, typename Tval, typename Op, typename HelperOp>
void test_reduce_by_key_const(ExPolicy && policy, Tkey, Tval, bool benchmark,
    const Op &op, const HelperOp &ho)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(Tval).name(), typeid(Op).name(), sync);
    std::cout << "\n";

    Tval rnd_min = -256;
    Tval rnd_max = 256;

    // vector of values, and keys
    std::vector<Tval> values, o_values;
    std::vector<Tkey> keys, o_keys;
    values.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);
    keys.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);

    std::vector<Tval> check_values;

    // use the default random engine and an uniform distribution for values
    std::mt19937 eng(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distr(rnd_min, rnd_max);

    // use the default random engine and an uniform distribution for keys
    std::mt19937 engk(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distrk(0, 256);

    // generate test data
    int keysize = 0;
    Tkey key = 0, helperkey = 0, lastkey = 0;
    for (/* */; keysize < HPX_REDUCE_BY_KEY_TEST_SIZE;)
        {
        do {
            key = static_cast<Tkey>(distrk(engk));
        } while (ho(key) == lastkey);
        helperkey = ho(key);
        lastkey = helperkey;
        //
        int numkeys = static_cast<int>(distrk(engk)) + 1;
        //
        Tval sum = 0;
        for (int i = 0; i < numkeys && keysize < HPX_REDUCE_BY_KEY_TEST_SIZE; ++i) {
            Tval value = static_cast<Tval>(distr(eng));
            keys.push_back(key);
            values.push_back(value);
            sum += value;
            keysize++;
        }
        check_values.push_back(sum);
    }
    o_values = values;
    o_keys = keys;

    const std::vector<Tkey> const_keys(keys.begin(), keys.end());
    const std::vector<Tval> const_values(values.begin(), values.end());

    hpx::util::high_resolution_timer t;
    // reduce_by_key, blocking when seq, par, par_vec
    auto result = hpx::parallel::reduce_by_key(
        std::forward<ExPolicy>(policy),
        const_keys.begin(), const_keys.end(),
        const_values.begin(),
        keys.begin(),
        values.begin(),
        op);
    double elapsed = t.elapsed();

    bool is_equal = std::equal(values.begin(), result.second, check_values.begin(),
        almost_equal());
    HPX_TEST(is_equal);
    if (is_equal) {
        if (benchmark) {
            // CDash graph plotting
            hpx::util::print_cdash_timing("ReduceByKeyTime", elapsed);
        }
    }
    else {
//         debug::output("keys     ", o_keys);
//         debug::output("values   ", o_values);
//         debug::output("key range", keys.begin(), result.first);
//         debug::output("val range", values.begin(), result.second);
//         debug::output("expected ", check_values);
//         throw std::string("Problem");
#if defined(EXTRA_DEBUG)
        for (std::size_t i = 0; i != check_values.size(); ++i)
        {
            if (values[i] != check_values[i])
            {
                std::cout
                    << i << ": "
                    << values[i] << " != " << check_values[i]
                    << "\n";
            }
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
template<typename ExPolicy, typename Tkey, typename Tval, typename Op, typename HelperOp>
void test_reduce_by_key_async(ExPolicy && policy, Tkey, Tval, const Op &op,
    const HelperOp &ho)
    {
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(Tval).name(), typeid(Op).name(), async);
    std::cout << "\n";

    Tval rnd_min = -256;
    Tval rnd_max = 256;

    // vector of values, and keys
    std::vector<Tval> values, o_values;
    std::vector<Tkey> keys, o_keys;
    values.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);
    keys.reserve(HPX_REDUCE_BY_KEY_TEST_SIZE);

    std::vector<Tval> check_values;

    // use the default random engine and an uniform distribution for values
    std::mt19937 eng(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distr(rnd_min, rnd_max);

    // use the default random engine and an uniform distribution for keys
    std::mt19937 engk(static_cast<unsigned int>(gen()));
    std::uniform_real_distribution<double> distrk(0, 256);

    // generate test data
    int keysize = 0;
    Tkey key = 0, helperkey = 0, lastkey = 0;
    for (/* */; keysize < HPX_REDUCE_BY_KEY_TEST_SIZE;)
        {
        do {
            key = static_cast<Tkey>(distrk(engk));
        } while (ho(key) == lastkey);
        helperkey = ho(key);
        lastkey = helperkey;
        //
        int numkeys = static_cast<Tkey>(distrk(engk)) + 1;
        //
        Tval sum = 0;
        for (int i = 0; i < numkeys && keysize < HPX_REDUCE_BY_KEY_TEST_SIZE; ++i) {
            Tval value = static_cast<Tval>(distr(eng));
            keys.push_back(key);
            values.push_back(value);
            sum += value;
            keysize++;
        }
        check_values.push_back(sum);
    }
    o_values = values;
    o_keys = keys;

    // reduce_by_key, blocking when seq, par, par_vec
    hpx::util::high_resolution_timer t;
    auto fresult = hpx::parallel::reduce_by_key(
        std::forward<ExPolicy>(policy),
        keys.begin(), keys.end(),
        values.begin(),
        keys.begin(),
        values.begin(),
        op);
    double async_seconds = t.elapsed();
    auto result = fresult.get();
    double sync_seconds= t.elapsed();

    std::cout << "Async time " << async_seconds << " Sync time " << sync_seconds << "\n";
    bool is_equal = std::equal(values.begin(), result.second, check_values.begin(),
        almost_equal());
    HPX_TEST(is_equal);
    if (is_equal) {
        //std::cout << "Test Passed\n";
    }
    else {
//         debug::output("keys     ", o_keys);
//         debug::output("values   ", o_values);
//         debug::output("key range", keys.begin(), result.first);
//         debug::output("val range", values.begin(), result.second);
//         debug::output("expected ", check_values);
//         throw std::string("Problem");
#if defined(EXTRA_DEBUG)
        for (std::size_t i = 0; i != check_values.size(); ++i)
        {
            if (values[i] != check_values[i])
            {
                std::cout
                    << i << ": "
                    << values[i] << " != " << check_values[i]
                    << "\n";
            }
        }
#endif
    }
}

////////////////////////////////////////////////////////////////////////////////
void test_reduce_by_key1()
{
    using namespace hpx::parallel;
    //
    hpx::util::high_resolution_timer t;
    do {
        test_reduce_by_key1(execution::seq, int(), int(), false,
            std::equal_to<int>(),
            [](int key) {return key;});
        test_reduce_by_key1(execution::par, int(), int(), false,
            std::equal_to<int>(),
            [](int key) {return key;});
        test_reduce_by_key1(execution::par_unseq, int(), int(), false,
            std::equal_to<int>(),
            [](int key) {return key;});
        //
        // default comparison operator (std::equal_to)
        test_reduce_by_key1(execution::seq, int(), double(), false,
            almost_equal(),
            [](int key) {return key;});
        test_reduce_by_key1(execution::par, int(), double(), false,
            almost_equal(),
            [](int key) {return key;});
        test_reduce_by_key1(execution::par_unseq, int(), double(), false,
            almost_equal(),
            [](int key) {return key;});
        //
        //
        test_reduce_by_key1(execution::seq, double(), double(), false,
            [](double a, double b) {
                return std::abs(std::floor(a) - std::floor(b)) < 1e-15;}, //-V550
            [](double a) {return std::floor(a);}
            );
        test_reduce_by_key1(execution::par, double(), double(), false,
            [](double a, double b) {
                return std::abs(std::floor(a) - std::floor(b)) < 1e-15;}, //-V550
            [](double a) {return std::floor(a);}
            );
        test_reduce_by_key1(execution::par_unseq, double(), double(), false,
            [](double a, double b) {
                return std::abs(std::floor(a) - std::floor(b)) < 1e-15;}, //-V550
            [](double a) {return std::floor(a);}
            );
    } while (t.elapsed() < 2);
    //
    hpx::util::high_resolution_timer t3;
    do {
        test_reduce_by_key_const(execution::seq, int(), int(), false,
            std::equal_to<int>(),
            [](int key) {return key;});
        //
        // default comparison operator (std::equal_to)
        test_reduce_by_key_const(execution::seq, int(), double(), false,
            almost_equal(),
            [](int key) {return key;});
        //
        test_reduce_by_key_const(execution::seq, double(), double(), false,
            [](double a, double b) {
                return std::abs(std::floor(a) - std::floor(b)) < 1e-15;}, //-V550
            [](double a) {return std::floor(a);}
            );
    } while (t3.elapsed() < 0.5);
    //
    hpx::util::high_resolution_timer t2;
    do {
        test_reduce_by_key_async(execution::seq(execution::task), int(), int(),
            std::equal_to<int>(),
            [](int key) {return key;});
        test_reduce_by_key_async(execution::par(execution::task), int(), int(),
            std::equal_to<int>(),
            [](int key) {return key;});
        //
        test_reduce_by_key_async(execution::seq(execution::task), int(), double(),
            almost_equal(),
            [](int key) {return key;});
        test_reduce_by_key_async(execution::par(execution::task), int(), double(),
            almost_equal(),
            [](int key) {return key;});
    } while (t2.elapsed() < 2);

    // one last test with timing output enabled
    test_reduce_by_key1(execution::par, double(), double(), true,
        [](double a, double b) {
            return std::abs(std::floor(a) - std::floor(b)) < 1e-15;}, //-V550
        [](double a) {return std::floor(a);}
        );
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    test_reduce_by_key1();
//    test_reduce_by_key2();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description
    desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
