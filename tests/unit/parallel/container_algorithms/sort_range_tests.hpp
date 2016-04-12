//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_IS_SORTED_MAY28_15_1320)
#define HPX_PARALLEL_TEST_IS_SORTED_MAY28_15_1320

//
#include <iomanip>
#include <limits>
#include <random>
#include <string>
//
#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_container_algorithm.hpp>
#include <hpx/util/lightweight_test.hpp>
//
#include <boost/range/functions.hpp>
#include <boost/format.hpp>
//
#include "test_utils.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#if !defined(HPX_SORT_TEST_SIZE_STRINGS)
#define HPX_SORT_TEST_SIZE_STRINGS 1000000
#endif

#if !defined(HPX_SORT_TEST_SIZE)
#define HPX_SORT_TEST_SIZE 5000000
#endif

// --------------------------------------------------------------------
// Fill a vector with random numbers in the range [lower, upper]
template <typename T>
void rnd_fill(std::vector<T> &V, const T lower, const T upper, const T seed)
{
    // use the default random engine and an uniform distribution
    boost::random::mt19937 eng(static_cast<unsigned int>(seed));
    boost::random::uniform_real_distribution<double> distr(lower, upper);

    for (auto &elem : V) {
        elem = static_cast<T>(distr(eng));
    }
}

// --------------------------------------------------------------------
// generate a random string of a given length
std::string random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

// --------------------------------------------------------------------
// fill a vector with random strings
void rnd_strings(std::vector<std::string> &V) {
    const std::size_t test_size = HPX_SORT_TEST_SIZE_STRINGS;
    // Fill vector with random strings
    V.clear();
    V.reserve(test_size);
    // random strings up to 128 chars long
    for (std::size_t i=0; i<test_size; i++) {
        V.push_back(random_string( std::rand() % 128));
    }
}

// --------------------------------------------------------------------
// check that the array is sorted correctly
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
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, random);

    // Fill vector with random values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy), c);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, std::less<T>(), elapsed, true) != 0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// call sort with a comparison operator
template <typename ExPolicy, typename T, typename Compare = std::less<T>>
        void test_sort1_comp(ExPolicy && policy, T, Compare comp = Compare())
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        sync, random);

    // Fill vector with random values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy), c, comp);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// async sort
template <typename ExPolicy, typename T, typename Compare = std::less<T>>
        void test_sort1_async(ExPolicy && policy, T, Compare comp = Compare())
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        async, random);

    // Fill vector with random values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, non blocking
    hpx::future<void> f =
        hpx::parallel::sort(std::forward<ExPolicy>(policy), c, comp);
    f.get();
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

// test exceptions
template <typename ExPolicy, typename T>
void test_sort_exception(ExPolicy && policy, T)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, random);

    // Fill vector with random values
    std::vector<T> c(5000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::parallel::sort(
                std::forward<ExPolicy>(policy),
                boost::make_iterator_range(
                    decorated_iterator(
                        c.begin(), [](){ throw std::runtime_error("test"); }),
                    decorated_iterator(c.end())));

            HPX_TEST(false);
        }
        catch(hpx::exception_list const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        if (caught_exception)
            std::cout << "OK, ";
        else
            std::cout << "Failed, ";
    }

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::parallel::sort(
                std::forward<ExPolicy>(policy),
                boost::make_iterator_range(
                    decorated_iterator(
                        c.begin(), [](){ throw std::bad_alloc(); }),
                    decorated_iterator(c.end())));

            HPX_TEST(false);
        }
        catch(std::bad_alloc const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        if (caught_exception)
            std::cout << "OK " << std::endl;
        else
            std::cout << "Failed " << std::endl;
    }
}

template <typename ExPolicy, typename T, typename Compare>
void test_sort_exception(ExPolicy && policy, T, Compare comp)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        sync, random);

    // Fill vector with random values
    std::vector<T> c(5000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::parallel::sort(
                std::forward<ExPolicy>(policy),
                boost::make_iterator_range(
                    decorated_iterator(
                        c.begin(), [](){ throw std::runtime_error("test"); }),
                    decorated_iterator(c.end())),
                    comp);

            HPX_TEST(false);
        }
        catch(hpx::exception_list const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        if (caught_exception)
            std::cout << "OK, ";
        else
            std::cout << "Failed, ";
    }

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::parallel::sort(
                std::forward<ExPolicy>(policy),
                boost::make_iterator_range(
                    decorated_iterator(
                        c.begin(), [](){ throw std::bad_alloc(); }),
                    decorated_iterator(c.end())),
                    comp);

            HPX_TEST(false);
        }
        catch(std::bad_alloc const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        if (caught_exception)
            std::cout << "OK " << std::endl;
        else
            std::cout << "Failed " << std::endl;
    }
}

// test exceptions
template <typename ExPolicy, typename T>
void test_sort_exception_async(ExPolicy && policy, T)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", async, random);

    // Fill vector with random values
    std::vector<T> c(5000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::future<void> f =
                hpx::parallel::sort(
                    std::forward<ExPolicy>(policy),
                    boost::make_iterator_range(
                        decorated_iterator(
                            c.begin(), [](){ throw std::runtime_error("test"); }),
                        decorated_iterator(c.end())));

            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch(hpx::exception_list const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
        if (caught_exception && returned_from_algorithm)
            std::cout << "OK, ";
        else
            std::cout << "Failed, ";
    }

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::future<void> f =
                hpx::parallel::sort(
                    std::forward<ExPolicy>(policy),
                    boost::make_iterator_range(
                        decorated_iterator(
                            c.begin(), [](){ throw std::bad_alloc(); }),
                        decorated_iterator(c.end())));

            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch(std::bad_alloc const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
        if (caught_exception && returned_from_algorithm)
            std::cout << "OK " << std::endl;
        else
            std::cout << "Failed " << std::endl;
    }
}

template <typename ExPolicy, typename T, typename Compare>
void test_sort_exception_async(ExPolicy && policy, T, Compare comp)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        async, random);

    // Fill vector with random values
    std::vector<T> c(5000);
    rnd_fill<T>(c, (std::numeric_limits<T>::min)(),
        (std::numeric_limits<T>::max)(), T(std::rand()));

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::future<void> f =
                hpx::parallel::sort(
                    std::forward<ExPolicy>(policy),
                    boost::make_iterator_range(
                        decorated_iterator(
                            c.begin(), [](){ throw std::runtime_error("test"); }),
                        decorated_iterator(c.end())),
                        comp);

            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch(hpx::exception_list const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
        if (caught_exception && returned_from_algorithm)
            std::cout << "OK, ";
        else
            std::cout << "Failed, ";
    }

    {
        // sort, blocking when seq, par, par_vec
        bool caught_exception = false;
        bool returned_from_algorithm = false;
        try {
            typedef typename std::vector<T>::iterator base_iterator;
            typedef test::decorated_iterator<
                    base_iterator, std::random_access_iterator_tag
                > decorated_iterator;

            hpx::future<void> f =
                hpx::parallel::sort(
                    std::forward<ExPolicy>(policy),
                    boost::make_iterator_range(
                        decorated_iterator(
                            c.begin(), [](){ throw std::bad_alloc(); }),
                        decorated_iterator(c.end())),
                        comp);

            returned_from_algorithm = true;
            f.get();

            HPX_TEST(false);
        }
        catch(std::bad_alloc const&) {
            caught_exception = true;
        }
        catch(...) {
            HPX_TEST(false);
        }

        HPX_TEST(caught_exception);
        HPX_TEST(returned_from_algorithm);
        if (caught_exception && returned_from_algorithm)
            std::cout << "OK " << std::endl;
        else
            std::cout << "Failed " << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
// already sorted
template <typename ExPolicy, typename T>
void test_sort2(ExPolicy && policy, T)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), "default", sync, sorted);

    // Fill vector with increasing values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
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
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        sync, sorted);

    // Fill vector with increasing values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
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
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(T).name(), typeid(Compare).name(),
        async, sorted);

    // Fill vector with random values
    std::vector<T> c(HPX_SORT_TEST_SIZE);
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

////////////////////////////////////////////////////////////////////////////////
// overload of test routine 1 for strings
// call sort on a string array with no comparison operator
template <typename ExPolicy>
void test_sort1(ExPolicy && policy, const std::string &)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(std::string).name(), "default", sync, random);

    // Fill vector with random strings
    std::vector<std::string> c;
    rnd_strings(c);

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end());
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, std::less<std::string>(), elapsed, true) != 0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// overload of test routine 1 for strings
// call sort on a string array with a comparison operator
template <typename ExPolicy, typename Compare = std::less<std::string>>
        void test_sort1_comp(ExPolicy && policy, const std::string &,
        Compare comp = Compare())
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(std::string).name(), typeid(Compare).name(),
        sync, random);

    // Fill vector with random strings
    std::vector<std::string> c;
    rnd_strings(c);

    boost::uint64_t t = hpx::util::high_resolution_clock::now();
    // sort, blocking when seq, par, par_vec
    hpx::parallel::sort(std::forward<ExPolicy>(policy),
            c.begin(), c.end(), comp);
    boost::uint64_t elapsed = hpx::util::high_resolution_clock::now() - t;

    bool is_sorted = (verify(c, comp, elapsed, true)!=0);
    HPX_TEST(is_sorted);
}

////////////////////////////////////////////////////////////////////////////////
// overload of test routine 1 for strings
// async sort
template <typename ExPolicy, typename Compare = std::less<std::string>>
        void test_sort1_async_string(ExPolicy && policy, const std::string &,
        Compare comp = Compare())
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");
    msg(typeid(ExPolicy).name(), typeid(std::string).name(), typeid(Compare).name(),
        async, random);

    // Fill vector with random strings
    std::vector<std::string> c;
    rnd_strings(c);

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
