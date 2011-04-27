////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2009 Helge Bahmann
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <boost/config.hpp>
#include <boost/atomic.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_pointer.hpp>

#include <hpx/util/lightweight_test.hpp>

using boost::atomic;
using boost::atomic_flag;
using boost::atomic_thread_fence;
using boost::memory_order_seq_cst;
using boost::memory_order_acquire;

using boost::is_pointer;
using boost::enable_if;
using boost::disable_if;

using hpx::util::report_errors;

namespace {

template <typename T>
void test_atomic_arithmetic(void)
{
    atomic<T> i(T(41));
    
    T n;
    
    HPX_TEST_LTE(sizeof(n), sizeof(i));
    
    bool success;
    
    n=i++;
    HPX_TEST_EQ(i, T(42));
    HPX_TEST_EQ(n, T(41));
    
    n=i--;
    HPX_TEST_EQ(n, T(42));
    HPX_TEST_EQ(i, T(41));
    
    n=++i;
    HPX_TEST_EQ(i, T(42));
    HPX_TEST_EQ(n, T(42));
    
    n=--i;
    HPX_TEST_EQ(n, T(41));
    HPX_TEST_EQ(i, T(41));
    
    n=i.fetch_and(T(15));
    HPX_TEST_EQ(n, T(41));
    HPX_TEST_EQ(i, T(9));
    
    n=i.fetch_or(T(17));
    HPX_TEST_EQ(n, T(9));
    HPX_TEST_EQ(i, T(25));
    
    n=i.fetch_xor(T(3));
    HPX_TEST_EQ(n, T(25));
    HPX_TEST_EQ(i, T(26));
    
    n=i.exchange(T(12));
    HPX_TEST_EQ(n, T(26));
    HPX_TEST_EQ(i, T(12));
    
    n=T(12);
    success=i.compare_exchange_strong(n, T(17));
    HPX_TEST(success);
    HPX_TEST_EQ(n, T(12));
    HPX_TEST_EQ(i, T(17));
    
    n=T(12);
    success=i.compare_exchange_strong(n, T(19));
    HPX_TEST(!success);
    HPX_TEST_EQ(n, T(17));
    HPX_TEST_EQ(i, T(17));
}

template <typename T>
typename enable_if<is_pointer<T>, T>::type
integral_cast(std::ptrdiff_t x)
{ return reinterpret_cast<T>(x); }

template <typename T>
typename disable_if<is_pointer<T>, T>::type
integral_cast(std::ptrdiff_t x)
{ return static_cast<T>(x); }

template<typename T>
void test_atomic_base(void)
{
    atomic<T> i;
    T n;
    
    HPX_TEST_LTE(sizeof(n), sizeof(i));
    
    bool success;
    
    i.store(integral_cast<T>(0));
    n=integral_cast<T>(40);
    success=i.compare_exchange_strong(n, integral_cast<T>(44));
    HPX_TEST(!success);
    HPX_TEST_EQ(n, integral_cast<T>(0));
    HPX_TEST_EQ(i.load(), integral_cast<T>(0));
    
    n=integral_cast<T>(0);
    success=i.compare_exchange_strong(n, integral_cast<T>(44));
    HPX_TEST(success);
    HPX_TEST_EQ(n, integral_cast<T>(0));
    HPX_TEST_EQ(i.load(), integral_cast<T>(44));
    
    n=i.exchange(integral_cast<T>(20));
    HPX_TEST_EQ(n, integral_cast<T>(44));
    HPX_TEST_EQ(i.load(), integral_cast<T>(20));
}

template<typename T>
void test_atomic_ptr(void)
{
    test_atomic_base<T*>();
    
    T array[10], *p;
    atomic<T*> ptr;
    
    ptr=&array[0];
    
    p=ptr++;
    HPX_TEST_EQ(p, &array[0]);
    HPX_TEST(ptr==&array[1]);
    p=++ptr;
    HPX_TEST_EQ(p, &array[2]);
    HPX_TEST(ptr==&array[2]);
    
    p=ptr.fetch_add(4);
    HPX_TEST_EQ(p, &array[2]);
    HPX_TEST(ptr==&array[6]);
    
    p=ptr.fetch_sub(4);
    HPX_TEST_EQ(p, &array[6]);
    HPX_TEST(ptr==&array[2]);
    
    p=ptr--;
    HPX_TEST_EQ(p, &array[2]);
    HPX_TEST(ptr==&array[1]);
    p=--ptr;
    HPX_TEST_EQ(p, &array[0]);
    HPX_TEST(ptr==&array[0]);
}

template<>
void test_atomic_base<bool>(void)
{
    atomic<bool> i;
    bool n;
    
    HPX_TEST_LTE(sizeof(n), sizeof(i));
    
    bool success;
    
    i=false;
    n=true;
    success=i.compare_exchange_strong(n, true);
    HPX_TEST(!success);
    HPX_TEST_EQ(n, false);
    HPX_TEST_EQ(i, false);
    
    n=false;
    success=i.compare_exchange_strong(n, true);
    HPX_TEST(success);
    HPX_TEST_EQ(n, false);
    HPX_TEST_EQ(i, true);
    
    n=i.exchange(false);
    HPX_TEST_EQ(n, true);
    HPX_TEST_EQ(i, false);
}

void test_atomic_flag()
{
    atomic_flag f(0);
    
    HPX_TEST(!f.test_and_set());
    HPX_TEST(f.test_and_set());
    f.clear();
    HPX_TEST(!f.test_and_set());
}

struct compound
{
    int i;

    inline bool operator==(const compound &c) const
    { return i==c.i; }
};

void test_atomic_struct(void)
{
    atomic<compound> i;
    compound n;
    
    compound zero={0}, one={1}, two={2};
    
    HPX_TEST_LTE(sizeof(n), sizeof(i));
    
    bool success;
    
    i.store(zero);
    n=one;
    success=i.compare_exchange_strong(n, two);
    HPX_TEST(!success);
    HPX_TEST(n==zero);
    HPX_TEST(i.load()==zero);
    
    n=zero;
    success=i.compare_exchange_strong(n, two);
    HPX_TEST(success);
    HPX_TEST(n==zero);
    HPX_TEST(i.load()==two);
    
    n=i.exchange(one);
    HPX_TEST(n==two);
    HPX_TEST(i.load()==one);
}

enum enum_type
{
    foo, bar
};

void test_fence()
{
    atomic_thread_fence(memory_order_acquire);
}

}

int main()
{
    test_atomic_arithmetic<char>();
    test_atomic_arithmetic<signed char>();
    test_atomic_arithmetic<unsigned char>();
    test_atomic_arithmetic<uint8_t>();
    test_atomic_arithmetic<int8_t>();
    test_atomic_arithmetic<short>();
    test_atomic_arithmetic<unsigned short>();
    test_atomic_arithmetic<uint16_t>();
    test_atomic_arithmetic<int16_t>();
    test_atomic_arithmetic<int>();
    test_atomic_arithmetic<unsigned int>();
    test_atomic_arithmetic<uint32_t>();
    test_atomic_arithmetic<int32_t>();
    test_atomic_arithmetic<long>();
    test_atomic_arithmetic<unsigned long>();
    test_atomic_arithmetic<uint64_t>();
    test_atomic_arithmetic<int64_t>();
    test_atomic_arithmetic<long long>();
    test_atomic_arithmetic<unsigned long long>();
    
    test_atomic_struct();
    
    test_atomic_base<void *>();
    test_atomic_ptr<int>();
    test_atomic_base<bool>();
    test_atomic_base<enum_type>();
    
    atomic_thread_fence(memory_order_seq_cst);
    
    test_fence();
    
    test_atomic_flag();

    return report_errors();
}

