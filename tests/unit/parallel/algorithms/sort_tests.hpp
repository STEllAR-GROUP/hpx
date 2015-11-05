//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_IS_SORTED_MAY28_15_1320)
#define HPX_PARALLEL_TEST_IS_SORTED_MAY28_15_1320

//
#include <random>
#include <limits>
#include <iomanip>
//
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <boost/range/functions.hpp>
#include <boost/format.hpp>
//
#include "test_utils.hpp"

#include  <boost/nondet_random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

// Fill a vector with random numbers in the range [lower, upper]
template <typename T>
void rnd_fill(std::vector<T> &V, const T lower, const T upper, const T seed)
{
    // use the default random engine and an uniform distribution
    std::default_random_engine eng(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<double> distr(lower, upper);

    for (auto &elem : V) {
        elem = static_cast<T>(distr(eng));
    }
}

template <class IA, typename Compare>
int verify(const std::vector <IA> &A, Compare comp, boost::uint64_t elapsed,
    bool print)
{
    if (A.size()<2) {
        // skip checks as we must be sorted
    }
    else {
        IA temp = *(A.begin());
        for (typename std::vector<IA>::const_iterator it=A.begin();
            it != A.end(); ++it)
        {
            if (comp((*it), temp)) {
                if (print) std::cout << "fail "
                  << boost::format("%8.6f") % (elapsed / 1e9)
                  << A.size() << std::endl;
                return 0;
            }
            temp = (*it);
        }
    }
    if (print) std::cout
      << "OK "
      << boost::format("%8.6f") % (elapsed / 1e9)
      << A.size() << std::endl;
    return 1;
}

#define msg(a,b,c,d,e) \
        std::cout \
        << std::setw(60) << a << std::setw(12) <<  b \
        << std::setw(40) << c << std::setw(6)  << #d \
        << std::setw(8)  << #e << "\t";

////////////////////////////////////////////////////////////////////////////////
// call sort with no comparison operator
template <typename ExPolicy, typename T>
void test_sort1(ExPolicy && policy, T)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, random);

    boost::random::random_device rseed;
    boost::random::mt19937 gen(rseed());

    // Fill vector with random values
    std::vector<T> c(5000000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(gen()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end());
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, std::less<T>(), elapsed, true) != 0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// call sort with a comparison operator
template <typename ExPolicy, typename T, typename Compare = std::less<T>>
        void test_sort1_comp(ExPolicy && policy, T, Compare comp = Compare())
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(), sync, random);

    boost::random::random_device rseed;
    boost::random::mt19937 gen(rseed());

    // Fill vector with random values
    std::vector<T> c(5000000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(gen()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end(), comp);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T, typename Compare = std::less<T>>
        void test_sort1_async(ExPolicy && policy, T, Compare comp = Compare())
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(), async, random);

    boost::random::random_device rseed;
    boost::random::mt19937 gen(rseed());

    // Fill vector with random values
    std::vector<T> c(5000000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(gen()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, non blocking
    hpx::future<void> f = hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end(), comp);
    f.get();
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// already sorted
template <typename ExPolicy, typename T>
void test_sort2(ExPolicy && policy, T)
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, sorted);

    // Fill vector with increasing values
    std::vector<T> c(5000000);
    std::iota(boost::begin(c), boost::end(c), 0);

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end());
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, std::less<T>(), elapsed, true) != 0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T, typename Compare = std::less<T> >
void test_sort2_comp(ExPolicy && policy, T, Compare comp = Compare())
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(), sync, sorted);

    // Fill vector with increasing values
    std::vector<T> c(5000000);
    std::iota(boost::begin(c), boost::end(c), 0);

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end(), comp);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T, typename Compare = std::less<T> >
void test_sort2_async(ExPolicy && policy, T, Compare comp = Compare())
{
    BOOST_STATIC_ASSERT(hpx::parallel::is_execution_policy<ExPolicy>::value);
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(), async, sorted);

    // Fill vector with random values
    std::vector<T> c(5000000);
    std::iota(boost::begin(c), boost::end(c), T(0));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, non blocking
    hpx::future<void> f = hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end(), comp);
    f.get();
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

#endif
