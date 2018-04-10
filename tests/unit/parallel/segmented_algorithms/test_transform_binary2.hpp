//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(TEST_TRANSFORM_BINARY2_HPP)
#define TEST_TRANSFORM_BINARY2_HPP

#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/include/parallel_count.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename U>
struct pfo
{
    template <typename T>
    U operator()(T& val) const
    {
        return U((T(2)*val));
    }
};

template <typename U>
struct add
{
    template <typename T1, typename T2>
    U operator()(T1 const& v1, T2 const& v2) const
    {
        return U(v1 + v2);
    }
};

template <typename T>
struct cmp
{
    cmp(T const& val = T()) : value_(val) {}

    template <typename T_>
    bool operator()(T_ const& val) const
    {
        return val == value_;
    }

    T value_;

    template <typename Archive>
    void serialize(Archive& ar, unsigned version)
    {
        ar & value_;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T, typename U=T>
void verify_values(ExPolicy && policy, hpx::partitioned_vector<T> const& v,
    U const& val)
{
    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;

    std::size_t size = 0;

    const_iterator end = v.end();
    for (const_iterator it = v.begin(); it != end; ++it, ++size)
    {
        HPX_TEST_EQ(*it, val);
    }

    HPX_TEST_EQ(size, v.size());
}

template <typename ExPolicy, typename T, typename U=T>
void verify_values_count(ExPolicy && policy,
    hpx::partitioned_vector<T> const& v, U const& val)
{
    HPX_TEST_EQ(
        std::size_t(hpx::parallel::count(
            policy, v.begin(), v.end(), val)),
        v.size());
    HPX_TEST_EQ(
        std::size_t(hpx::parallel::count_if(
            policy, v.begin(), v.end(), cmp<T>(val))),
        v.size());
}

template <typename ExPolicy, typename T, typename U=T>
void verify_values_count_async(ExPolicy && policy,
    hpx::partitioned_vector<T> const& v, U const& val)
{
    HPX_TEST_EQ(
        std::size_t(hpx::parallel::count(
            policy, v.begin(), v.end(), val).get()),
        v.size());
    HPX_TEST_EQ(
        std::size_t(hpx::parallel::count_if(
            policy, v.begin(), v.end(), cmp<T>(val)).get()),
        v.size());
}

template <typename ExPolicy, typename T, typename U=T, typename V=T>
void test_transform_binary2(ExPolicy && policy, hpx::partitioned_vector<T>& v,
    hpx::partitioned_vector<U>& w, hpx::partitioned_vector<V>& x, V val)
{
    verify_values(policy, v, val);
    verify_values_count(policy, v, val);
    verify_values(policy, w, val);
    verify_values_count(policy, w, val);

    hpx::parallel::transform(policy, v.begin(), v.end(), w.begin(), w.end(),
        x.begin(), add<V>());

    verify_values(policy, x, 2*val);
    verify_values_count(policy, x, 2*val);
}

template <typename ExPolicy, typename T, typename U=T, typename V=T>
void test_transform_binary2_async(ExPolicy && policy,
    hpx::partitioned_vector<T>& v, hpx::partitioned_vector<U>& w,
    hpx::partitioned_vector<V>& x, V val)
{
    verify_values(policy, v, val);
    verify_values_count_async(policy, v, val);
    verify_values(policy, w, val);
    verify_values_count_async(policy, w, val);

    hpx::parallel::transform(policy, v.begin(), v.end(), w.begin(), w.end(),
        x.begin(), add<V>()).get();

    verify_values(policy, x, 2*val);
    verify_values_count_async(policy, x, 2*val);
}

template <typename T, typename U=T, typename V=T>
void transform_binary2_tests(std::vector<hpx::id_type> &localities)
{
    std::size_t const length = 12;
    {
        hpx::partitioned_vector<T> v;
        hpx::partitioned_vector<U> w;
        hpx::partitioned_vector<V> x;
        hpx::parallel::transform(hpx::parallel::execution::seq,
            v.begin(), v.end(), w.begin(), w.end(), x.begin(), add<V>());
        hpx::parallel::transform(hpx::parallel::execution::par,
            v.begin(), v.end(), w.begin(), w.end(), x.begin(), add<V>());
        hpx::parallel::transform(
            hpx::parallel::execution::seq(hpx::parallel::execution::task),
            v.begin(), v.end(), w.begin(), w.end(), x.begin(), add<V>()).get();
        hpx::parallel::transform(
            hpx::parallel::execution::par(hpx::parallel::execution::task),
            v.begin(), v.end(), w.begin(), w.end(), x.begin(), add<V>()).get();
    }

    {
        hpx::partitioned_vector<T> v(length, T(1),hpx::container_layout(localities));
        hpx::partitioned_vector<U> w(length, U(1),hpx::container_layout(localities));
        hpx::partitioned_vector<V> x(length,hpx::container_layout(localities));
        test_transform_binary2(hpx::parallel::execution::seq, v, w, x, V(1));
        test_transform_binary2(hpx::parallel::execution::par, v, w, x, V(1));
        test_transform_binary2_async(
            hpx::parallel::execution::seq(hpx::parallel::execution::task),
            v, w, x, V(1));
        test_transform_binary2_async(
            hpx::parallel::execution::par(hpx::parallel::execution::task),
            v, w, x, V(1));
    }
}

#endif
