#ifndef BOOST_DETAIL_ATOMIC_GCC_ARMV6P_HPP
#define BOOST_DETAIL_ATOMIC_GCC_ARMV6P_HPP

//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  Copyright (c) 2009 Helge Bahmann
//  Copyright (c) 2009 Phil Endecott
//  ARM Code by Phil Endecott, based on other architectures.


#include <boost/memory_order.hpp>
#include <boost/atomic/detail/base.hpp>
#include <boost/atomic/detail/builder.hpp>

// From the ARM Architecture Reference Manual for architecture v6:
// 
// LDREX{<cond>} <Rd>, [<Rn>]
// <Rd> Specifies the destination register for the memory word addressed by <Rd>
// <Rn> Specifies the register containing the address.
// 
// STREX{<cond>} <Rd>, <Rm>, [<Rn>]
// <Rd> Specifies the destination register for the returned status value.
//      0  if the operation updates memory
//      1  if the operation fails to update memory
// <Rm> Specifies the register containing the word to be stored to memory.
// <Rn> Specifies the register containing the address.
//
// ARM v7 is like ARM v6 plus:
// There are half-word and byte versions of the LDREX and STREX instructions,
// LDREXH, LDREXB, STREXH and STREXB.
// I think you can supply an immediate offset to the address.
//
// A memory barrier is effected using a "co-processor 15" instruction.

namespace boost {
namespace detail {
namespace atomic {


#define BOOST_ATOMIC_ARM_DMB "mcr\tp15, 0, r0, c7, c10, 5"

static inline void fence_before(memory_order order)
{
	// FIXME I don't understand enough about barriers to know what this should do.
	switch(order) {
		case memory_order_release:
		case memory_order_acq_rel:
		case memory_order_seq_cst:
			__asm__ __volatile__ (BOOST_ATOMIC_ARM_DMB ::: "memory");
		default:;
	}
}

static inline void fence_after(memory_order order)
{
	// FIXME I don't understand enough about barriers to know what this should do.
	switch(order) {
		case memory_order_acquire:
		case memory_order_acq_rel:
		case memory_order_seq_cst:
			__asm__ __volatile__ (BOOST_ATOMIC_ARM_DMB ::: "memory");
		case memory_order_consume:
			__asm__ __volatile__ ("" ::: "memory");
		default:;
	}
}

#undef BOOST_ATOMIC_ARM_DMB


template<typename T>
class atomic_arm_4 {
public:
	typedef T integral_type;
	explicit atomic_arm_4(T v) : i(v) {}
	atomic_arm_4() {}
	T load(memory_order order=memory_order_seq_cst) const volatile
	{
		T v=const_cast<volatile const T &>(i);
		fence_after(order);
		return v;
	}
	void store(T v, memory_order order=memory_order_seq_cst) volatile
	{
		fence_before(order);
		const_cast<volatile T &>(i)=v;
	}
        bool compare_exchange_weak(
                T &expected,
                T desired,
                memory_order success_order,
                memory_order failure_order) volatile
	{
		fence_before(success_order);
		int success;
		int tmp;
		__asm__ __volatile__(
			"mov     %1, #0\n"        // success = 0
			"ldrex   %0, [%3]\n"      // expected' = *(&i)
			"teq     %0, %4\n"        // flags = expected'==expected
			"strexeq %2, %5, [%3]\n"  // if (flags.equal) *(&i) = desired, tmp = !OK
			"teqeq   %2, #0\n"        // if (flags.equal) flags = tmp==0
			"moveq   %1, #1\n"        // if (flags.equal) success = 1
				: "=&r" (expected),  // %0
				  "=&r" (success),   // %1
				  "=&r" (tmp)        // %2
                                : "r" (&i),          // %3
				  "r" (expected),    // %4
				  "r" ((int)desired) // %5
				: "cc"
			);
                if (success) fence_after(success_order);
                else fence_after(failure_order);
		return success;
	}
	
	bool is_lock_free(void) const volatile {return true;}
protected:
	inline T fetch_add_var(T c, memory_order order) volatile
	{
		fence_before(order);
		T original, tmp;
		int tmp2;
		__asm__ __volatile__(
			"1: ldrex %0, [%3]\n"      // original = *(&i)
			"add      %1, %0, %4\n"    // tmp = original + c
			"strex    %2, %1, [%3]\n"  // *(&i) = tmp;  tmp2 = !OK
			"teq      %2, #0\n"        // flags = tmp2==0
			"bne      1b\n"            // if (!flags.equal) goto 1
				: "=&r" (original), // %0
				  "=&r" (tmp),      // %1
				  "=&r" (tmp2)      // %2
				: "r" (&i),         // %3
				  "r" (c)           // %4
				: "cc"
			);
		fence_after(order);
		return original;
	}
	inline T fetch_inc(memory_order order) volatile
	{
		fence_before(order);
		T original, tmp;
		int tmp2;
		__asm__ __volatile__(
			"1: ldrex %0, [%3]\n"      // original = *(&i)
			"add      %1, %0, #1\n"    // tmp = original + 1
			"strex    %2, %1, [%3]\n"  // *(&i) = tmp;  tmp2 = !OK
			"teq      %2, #0\n"        // flags = tmp2==0
			"bne      1b\n"            // if (!flags.equal) goto 1
				: "=&r" (original), // %0
				  "=&r" (tmp),      // %1
				  "=&r" (tmp2)      // %2
				: "r" (&i)          // %3
				: "cc"
			);
		fence_after(order);
		return original;
	}
	inline T fetch_dec(memory_order order) volatile
	{
		fence_before(order);
		T original, tmp;
		int tmp2;
		__asm__ __volatile__(
			"1: ldrex %0, [%3]\n"      // original = *(&i)
			"sub      %1, %0, #1\n"    // tmp = original - 1
			"strex    %2, %1, [%3]\n"  // *(&i) = tmp;  tmp2 = !OK
			"teq      %2, #0\n"        // flags = tmp2==0
			"bne      1b\n"            // if (!flags.equal) goto 1
				: "=&r" (original), // %0
				  "=&r" (tmp),      // %1
				  "=&r" (tmp2)      // %2
				: "r" (&i)          // %3
				: "cc"
			);
		fence_after(order);
		return original;
	}
private:
	T i;
};


// #ifdef _ARM_ARCH_7
// FIXME TODO can add native byte and halfword version here


template<typename T>
class platform_atomic_integral<T, 4> : public build_atomic_from_typical<build_exchange<atomic_arm_4<T> > > {
public:
	typedef build_atomic_from_typical<build_exchange<atomic_arm_4<T> > > super;
	explicit platform_atomic_integral(T v) : super(v) {}
	platform_atomic_integral(void) {}
};

template<typename T>
class platform_atomic_integral<T, 1>: public build_atomic_from_larger_type<atomic_arm_4<uint32_t>, T> {
public:
	typedef build_atomic_from_larger_type<atomic_arm_4<uint32_t>, T> super;
	
	explicit platform_atomic_integral(T v) : super(v) {}
	platform_atomic_integral(void) {}
};

template<typename T>
class platform_atomic_integral<T, 2>: public build_atomic_from_larger_type<atomic_arm_4<uint32_t>, T> {
public:
	typedef build_atomic_from_larger_type<atomic_arm_4<uint32_t>, T> super;
	
	explicit platform_atomic_integral(T v) : super(v) {}
	platform_atomic_integral(void) {}
};



typedef build_exchange<atomic_arm_4<void *> > platform_atomic_address;

}
}
}

#endif
