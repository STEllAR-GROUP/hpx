//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/traits/pending/is_callable.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <boost/config.hpp>

struct X { void operator()(int); };
struct Xc { void operator()(int) const; };

void nullary_function()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<void()>::value == true), "nullary function");
}

void lambdas()
{
    using hpx::traits::is_callable;
#   if !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_CXX11_DECLTYPE)
    auto lambda = [](){};
    HPX_TEST_MSG((is_callable<decltype(lambda)>::value == true), "lambda");
#   endif
}

void functions_byval_params()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<void(int), int>::value == true), "fun-value/value");
    HPX_TEST_MSG((is_callable<void(int), int&>::value == true), "fun-value/lvref");
    HPX_TEST_MSG((is_callable<void(int), int const&>::value == true), "fun-value/const-lvref");
    HPX_TEST_MSG((is_callable<void(int), BOOST_FWD_REF(int)>::value == true), "fun-value/rvref");
    HPX_TEST_MSG((is_callable<void(int), BOOST_FWD_REF(int const)>::value == true), "fun-value/const-rvref");
    HPX_TEST_MSG((is_callable<void(int const), int>::value == true), "fun-const-value/value");
    HPX_TEST_MSG((is_callable<void(int const), int&>::value == true), "fun-const-value/lvref");
    HPX_TEST_MSG((is_callable<void(int const), int const&>::value == true), "fun-const-value/const-lvref");
    HPX_TEST_MSG((is_callable<void(int const), BOOST_FWD_REF(int)>::value == true), "fun-const-value/rvref");
    HPX_TEST_MSG((is_callable<void(int const), BOOST_FWD_REF(int const)>::value == true), "fun-const-value/const-rvref");
}

void functions_bylvref_params()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<void(int&), int>::value == false), "fun-lvref/value");
    HPX_TEST_MSG((is_callable<void(int&), int&>::value == true), "fun-lvref/lvref");
    HPX_TEST_MSG((is_callable<void(int&), int const&>::value == false), "fun-lvref/const-lvref");
    HPX_TEST_MSG((is_callable<void(int&), BOOST_FWD_REF(int)>::value == false), "fun-lvref/rvref");
    HPX_TEST_MSG((is_callable<void(int&), BOOST_FWD_REF(int const)>::value == false), "fun-lvref/const-rvref");
    HPX_TEST_MSG((is_callable<void(int const&), int>::value == true), "fun-const-lvref/value");
    HPX_TEST_MSG((is_callable<void(int const&), int&>::value == true), "fun-const-lvref/lvref");
    HPX_TEST_MSG((is_callable<void(int const&), int const&>::value == true), "fun-const-lvref/const-lvref");
    HPX_TEST_MSG((is_callable<void(int const&), BOOST_FWD_REF(int)>::value == true), "fun-const-lvref/rvref");
    HPX_TEST_MSG((is_callable<void(int const&), BOOST_FWD_REF(int const)>::value == true), "fun-const-lvref/const-rvref");
}

void functions_byrvref_params()
{
    using hpx::traits::is_callable;

#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    HPX_TEST_MSG((is_callable<void(int&&), int>::value == true), "fun-rvref/value");
    HPX_TEST_MSG((is_callable<void(int&&), int&>::value == false), "fun-rvref/lvref");
    HPX_TEST_MSG((is_callable<void(int&&), int const&>::value == false), "fun-rvref/const-lvref");
    HPX_TEST_MSG((is_callable<void(int&&), BOOST_FWD_REF(int)>::value == true), "fun-rvref/rvref");
    HPX_TEST_MSG((is_callable<void(int&&), BOOST_FWD_REF(int const)>::value == false), "fun-rvref/const-rvref");
    HPX_TEST_MSG((is_callable<void(int const&&), int>::value == true), "fun-const-rvref/value");
    HPX_TEST_MSG((is_callable<void(int const&&), int&>::value == false), "fun-const-rvref/lvref");
    HPX_TEST_MSG((is_callable<void(int const&&), int const&>::value == false), "fun-const-rvref/const-lvref");
    HPX_TEST_MSG((is_callable<void(int const&&), BOOST_FWD_REF(int)>::value == true), "fun-const-rvref/rvref");
    HPX_TEST_MSG((is_callable<void(int const&&), BOOST_FWD_REF(int const)>::value == true), "fun-const-rvref/const-rvref");
#   endif
}

void member_function_pointers()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<int (X::*)(double), X*, float>::value == true), "mem-fun-ptr/ptr");
    HPX_TEST_MSG((is_callable<int (X::*)(double), X const*, float>::value == false), "mem-fun-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<int (X::*)(double), X&, float>::value == true), "mem-fun-ptr/lvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double), X const&, float>::value == false), "mem-fun-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double), BOOST_FWD_REF(X), float>::value == true), "mem-fun-ptr/rvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double), BOOST_FWD_REF(X const), float>::value == false), "mem-fun-ptr/const-rvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, X*, float>::value == true), "const-mem-fun-ptr/ptr");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, X const*, float>::value == true), "const-mem-fun-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, X&, float>::value == true), "const-mem-fun-ptr/lvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, X const&, float>::value == true), "const-mem-fun-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, BOOST_FWD_REF(X), float>::value == true), "const-mem-fun-ptr/rvref");
    HPX_TEST_MSG((is_callable<int (X::*)(double) const, BOOST_FWD_REF(X const), float>::value == true), "const-mem-fun-ptr/const-rvref");
}

void member_object_pointers()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<int (X::*), X*>::value == true), "mem-obj-ptr/ptr");
    HPX_TEST_MSG((is_callable<int (X::*), X const*>::value == true), "mem-obj-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<int (X::*), X&>::value == true), "mem-obj-ptr/lvref");
    HPX_TEST_MSG((is_callable<int (X::*), X const&>::value == true), "mem-obj-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<int (X::*), BOOST_FWD_REF(X)>::value == true), "mem-obj-ptr/rvref");
    HPX_TEST_MSG((is_callable<int (X::*), BOOST_FWD_REF(X const)>::value == true), "mem-obj-ptr/const-rvref");
}

void function_objects()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<X, int>::value == true), "fun-obj/value");
    HPX_TEST_MSG((is_callable<X const, int>::value == false), "fun-obj/const-value");
    HPX_TEST_MSG((is_callable<X*, int>::value == false), "fun-obj/ptr");
    HPX_TEST_MSG((is_callable<X const*, int>::value == false), "fun-obj/const-ptr");
    HPX_TEST_MSG((is_callable<X&, int>::value == true), "fun-obj/lvref");
    HPX_TEST_MSG((is_callable<X const&, int>::value == false), "fun-obj/const-lvref");
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    HPX_TEST_MSG((is_callable<X&&, int>::value == true), "fun-obj/rvref");
    HPX_TEST_MSG((is_callable<X const&&, int>::value == false), "fun-obj/const-rvref");
#   endif

    HPX_TEST_MSG((is_callable<Xc, int>::value == true), "const-fun-obj/value");
    HPX_TEST_MSG((is_callable<Xc const, int>::value == true), "const-fun-obj/const-value");
    HPX_TEST_MSG((is_callable<Xc*, int>::value == false), "const-fun-obj/ptr");
    HPX_TEST_MSG((is_callable<Xc const*, int>::value == false), "const-fun-obj/const-ptr");
    HPX_TEST_MSG((is_callable<Xc&, int>::value == true), "const-fun-obj/lvref");
    HPX_TEST_MSG((is_callable<Xc const&, int>::value == true), "const-fun-obj/const-lvref");
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    HPX_TEST_MSG((is_callable<Xc&&, int>::value == true), "const-fun-obj/rvref");
    HPX_TEST_MSG((is_callable<Xc const&&, int>::value == true), "const-fun-obj/const-rvref");
#   endif
}

///////////////////////////////////////////////////////////////////////////////
void run_local_tests()
{
    nullary_function();
    lambdas();
    functions_byval_params();
    functions_bylvref_params();
    functions_byrvref_params();
    member_function_pointers();
    member_object_pointers();
    function_objects();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // run local tests
    run_local_tests();
}
