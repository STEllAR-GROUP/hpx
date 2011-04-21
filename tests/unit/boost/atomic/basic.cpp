#include <typeinfo>
#include <boost/atomic.hpp>
#include <stdio.h>

#include <assert.h>

using namespace boost;

template<typename T>
void test_atomic_arithmetic(void)
{
	atomic<T> i(41);
	
	T n;
	
	printf("Type=%s, size=%ld, atomic_size=%ld, lockfree=%d\n",
		typeid(T).name(), (long)sizeof(n), (long)sizeof(i), i.is_lock_free());
	
	assert(sizeof(i)>=sizeof(n));
	
	bool success;
	
	n=i++;
	assert(i==42);
	assert(n==41);
	
	n=i--;
	assert(n==42);
	assert(i==41);
	
	n=++i;
	assert(i==42);
	assert(n==42);
	
	n=--i;
	assert(n==41);
	assert(i==41);
	
	n=i.fetch_and(15);
	assert(n==41);
	assert(i==9);
	
	n=i.fetch_or(17);
	assert(n==9);
	assert(i==25);
	
	n=i.fetch_xor(3);
	assert(n==25);
	assert(i==26);
	
	n=i.exchange(12);
	assert(n==26);
	assert(i==12);
	
	n=12;
	success=i.compare_exchange_strong(n, 17);
	assert(success);
	assert(n==12);
	assert(i==17);
	
	n=12;
	success=i.compare_exchange_strong(n, 19);
	assert(!success);
	assert(n==17);
	assert(i==17);
}

template<typename T>
void test_atomic_base(void)
{
	atomic<T> i;
	T n;
	
	printf("Type=%s, size=%ld, atomic_size=%ld, lockfree=%d\n",
		typeid(T).name(), (long)sizeof(n), (long)sizeof(i), i.is_lock_free());
	
	assert(sizeof(i)>=sizeof(n));
	
	bool success;
	
	i.store((T)0);
	n=(T)40;
	success=i.compare_exchange_strong(n, (T)44 /*boost::memory_order_relaxed*/);
	assert(!success);
	assert(n==(T)0);
	assert(i.load()==(T)0);
	
	n=(T)0;
	success=i.compare_exchange_strong(n, (T)44);
	assert(success);
	assert(n==(T)0);
	assert(i.load()==(T)44);
	
 	n=i.exchange((T)20);
	assert(n==(T)44);
	assert(i.load()==(T)20);
}

template<typename T>
void test_atomic_ptr(void)
{
	test_atomic_base<T *>();
	
	T array[10], *p;
	atomic<T *> ptr;
	
	ptr=&array[0];
	
	p=ptr++;
	assert(p==&array[0]);
	assert(ptr==&array[1]);
	p=++ptr;
	assert(p==&array[2]);
	assert(ptr==&array[2]);
	
	p=ptr.fetch_add(4);
	assert(p==&array[2]);
	assert(ptr==&array[6]);
	
	p=ptr.fetch_sub(4);
	assert(p==&array[6]);
	assert(ptr==&array[2]);
	
	p=ptr--;
	assert(p==&array[2]);
	assert(ptr==&array[1]);
	p=--ptr;
	assert(p==&array[0]);
	assert(ptr==&array[0]);
}

template<>
void test_atomic_base<bool>(void)
{
	atomic<bool> i;
	bool n;
	
	printf("Type=bool, size=%ld, atomic_size=%ld, lockfree=%d\n",
		(long)sizeof(n), (long)sizeof(i), i.is_lock_free());
	
	assert(sizeof(i)>=sizeof(n));
	
	bool success;
	
	i=false;
	n=true;
	success=i.compare_exchange_strong(n, true);
	assert(!success);
	assert(n==false);
	assert(i==false);
	
	n=false;
	success=i.compare_exchange_strong(n, true);
	assert(success);
	assert(n==false);
	assert(i==true);
	
	n=i.exchange(false);
	assert(n==true);
	assert(i==false);
}

void test_atomic_flag()
{
	atomic_flag f(0);
	
	assert(!f.test_and_set());
	assert(f.test_and_set());
	f.clear();
	assert(!f.test_and_set());
}

struct Compound {
	int i;
	
	inline bool operator==(const Compound &c) const {return i==c.i;}
};

void test_atomic_struct(void)
{
	atomic<Compound> i;
	Compound n;
	
	Compound zero={0}, one={1}, two={2};
	
	assert(sizeof(i)>=sizeof(n));
	
	bool success;
	
	i.store(zero);
	n=one;
	success=i.compare_exchange_strong(n, two);
	assert(!success);
	assert(n==zero);
	assert(i.load()==zero);
	
	n=zero;
	success=i.compare_exchange_strong(n, two);
	assert(success);
	assert(n==zero);
	assert(i.load()==two);
	
	n=i.exchange(one);
	assert(n==two);
	assert(i.load()==one);
}

enum TestEnum {
	Foo, Bar
};

void test_fence()
{
	atomic_thread_fence(memory_order_acquire);
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
	test_atomic_base<TestEnum>();
	
	atomic_thread_fence(memory_order_seq_cst);
	
	test_fence();
	
	test_atomic_flag();
}
