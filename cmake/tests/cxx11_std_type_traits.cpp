////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <type_traits>

struct callable
{
    callable() {}
    explicit callable(int) {}
    int operator()(){ return 0; }
};

enum enum_type
{
    enum_value = 0
};

union union_type
{
    int i;
    double d;
};

int main()
{
    using namespace std;

    int x = 0;
    add_const<int>::type* rc = &x;
    decay<int const&>::type* d = &x;
    result_of<callable()>::type* ro = &x;
    is_convertible<int, long>::type ic;
    is_constructible<callable, int>::type icc;

    true_type tt;
    false_type ft;
    integral_constant<bool, false>::type icb;
    bool_constant<false>::type bc;

    is_void<void>::type iv;
    is_null_pointer<nullptr_t>::type inp;
    is_integral<int>::type ii;
    is_floating_point<double>::type ifp;
    is_array<int[]>::type ia;
    is_enum<enum_type>::type ie;
    is_union<union_type>::type ut;
    is_class<callable>::type icall;
    is_function<void()>::type ift;
    is_pointer<int*>::type ip;
    is_lvalue_reference<int&>::type ilr;
    is_rvalue_reference<int&&>::type irr;
    is_member_object_pointer<int>::type imop;
    is_member_function_pointer<int>::type imfp;
    is_fundamental<int>::type ifund;
    is_arithmetic<int>::type iar;
    is_scalar<int>::type isc;
    is_object<int>::type io;
    is_compound<int>::type icomp;
    is_reference<int&>::type ir;
    is_member_pointer<int>::type imp;
    is_const<int const>::type iconst;
    is_volatile<int volatile>::type ivol;
    is_trivial<int>::type itr;
    is_trivially_copyable<int>::type icpy;
    is_standard_layout<int>::type ilay;
    is_pod<int>::type ipd;
    is_literal_type<int>::type ilit;
    is_empty<int>::type iempty;
    is_polymorphic<int>::type ipoly;
    is_abstract<int>::type iabst;
    is_signed<int>::type isign;
    is_unsigned<int>::type iusign;
    is_trivially_constructible<int>::type itctr;
    is_nothrow_constructible<int>::type intctr;
    is_default_constructible<int>::type idctr;
    is_trivially_default_constructible<int>::type itdctr;
    is_nothrow_default_constructible<int>::type indctr;
    is_copy_constructible<int>::type icctr;
    is_trivially_copy_constructible<int>::type itcctr;
    is_nothrow_copy_constructible<int>::type incctr;
    is_move_constructible<int>::type imctr;
    is_trivially_move_constructible<int>::type itmctr;
    is_nothrow_move_constructible<int>::type inmctr;
    is_assignable<int, double>::type iass;
    is_trivially_assignable<int, double>::type itass;
    is_nothrow_assignable<int, double>::type inass;
    is_copy_assignable<int>::type icass;
    is_trivially_copy_assignable<int>::type itcass;
    is_nothrow_copy_assignable<int>::type incass;
    is_move_assignable<int>::type imass;
    is_trivially_move_assignable<int>::type itmass;
    is_nothrow_move_assignable<int>::type inmass;
    is_destructible<int>::type idtr;
    is_trivially_destructible<int>::type itdtr;
    is_nothrow_destructible<int>::type indtr;
    alignment_of<int>::type algn;
    rank<int>::type rnk;
    extent<int>::type ext;
    is_same<int, double>::type same;
    is_base_of<int, double>::type ibo;
    remove_cv<int const>::type rcv;
    remove_const<int const>::type rconst;
    remove_volatile<int volatile>::type rv;
    add_cv<int>::type acv = 0;
    add_volatile<int>::type av;
    remove_reference<int&>::type rref;
    add_lvalue_reference<int>::type alref = x;
    add_rvalue_reference<int>::type arref = int();
    remove_pointer<int*>::type rptr;
    add_pointer<int>::type aptr;
    make_signed<unsigned int>::type msign;
    make_unsigned<int>::type musign;
    remove_extent<int[1]>::type rext;
    remove_all_extents<int[1]>::type raext;
    aligned_storage<1, 2>::type aligns;
    enable_if<true, int>::type eif;
    conditional<true, int, double>::type cond;
    underlying_type<enum_type>::type utyp;
}

