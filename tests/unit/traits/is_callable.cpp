//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/traits/is_callable.hpp>

#include <hpx/util/lightweight_test.hpp>

struct X { void operator()(int); };
struct Xc { void operator()(int) const; };

void nullary_function()
{
    using hpx::traits::is_callable;

    typedef void (*f)();
    HPX_TEST_MSG((is_callable<f()>::value == true), "nullary function");
}

void lambdas()
{
    using hpx::traits::is_callable;
#   if !defined(BOOST_NO_CXX11_LAMBDAS) && !defined(BOOST_NO_CXX11_DECLTYPE)
    auto lambda = [](){};

    typedef decltype(lambda) f;
    HPX_TEST_MSG((is_callable<f()>::value == true), "lambda");
#   endif
}

void functions_byval_params()
{
    using hpx::traits::is_callable;

    typedef void (*f)(int);
    HPX_TEST_MSG((is_callable<f(int)>::value == true), "fun-value/value");
    HPX_TEST_MSG((is_callable<f(int&)>::value == true), "fun-value/lvref");
    HPX_TEST_MSG((is_callable<f(int const&)>::value == true), "fun-value/const-lvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int))>::value == true), "fun-value/rvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int const))>::value == true), "fun-value/const-rvref");

    typedef void (*fc)(int const);
    HPX_TEST_MSG((is_callable<fc(int)>::value == true), "fun-const-value/value");
    HPX_TEST_MSG((is_callable<fc(int&)>::value == true), "fun-const-value/lvref");
    HPX_TEST_MSG((is_callable<fc(int const&)>::value == true), "fun-const-value/const-lvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int))>::value == true), "fun-const-value/rvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int const))>::value == true), "fun-const-value/const-rvref");
}

void functions_bylvref_params()
{
    using hpx::traits::is_callable;

    typedef void (*f)(int&);
    HPX_TEST_MSG((is_callable<f(int)>::value == false), "fun-lvref/value");
    HPX_TEST_MSG((is_callable<f(int&)>::value == true), "fun-lvref/lvref");
    HPX_TEST_MSG((is_callable<f(int const&)>::value == false), "fun-lvref/const-lvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int))>::value == false), "fun-lvref/rvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int const))>::value == false), "fun-lvref/const-rvref");

    typedef void (*fc)(int const&);
    HPX_TEST_MSG((is_callable<fc(int)>::value == true), "fun-const-lvref/value");
    HPX_TEST_MSG((is_callable<fc(int&)>::value == true), "fun-const-lvref/lvref");
    HPX_TEST_MSG((is_callable<fc(int const&)>::value == true), "fun-const-lvref/const-lvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int))>::value == true), "fun-const-lvref/rvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int const))>::value == true), "fun-const-lvref/const-rvref");
}

void functions_byrvref_params()
{
    using hpx::traits::is_callable;

#ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    typedef void (*f)(int&&);
    HPX_TEST_MSG((is_callable<f(int)>::value == true), "fun-rvref/value");
#   if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 40500
    HPX_TEST_MSG((is_callable<f(int&)>::value == false), "fun-rvref/lvref");
#   endif
    HPX_TEST_MSG((is_callable<f(int const&)>::value == false), "fun-rvref/const-lvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int))>::value == true), "fun-rvref/rvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(int const))>::value == false), "fun-rvref/const-rvref");

    typedef void (*fc)(int const&&);
    HPX_TEST_MSG((is_callable<fc(int)>::value == true), "fun-const-rvref/value");
#   if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 40500
    HPX_TEST_MSG((is_callable<fc(int&)>::value == false), "fun-const-rvref/lvref");
    HPX_TEST_MSG((is_callable<fc(int const&)>::value == false), "fun-const-rvref/const-lvref");
#   endif
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int))>::value == true), "fun-const-rvref/rvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(int const))>::value == true), "fun-const-rvref/const-rvref");
#endif
}

void member_function_pointers()
{
    using hpx::traits::is_callable;

    typedef int (X::*f)(double);
    HPX_TEST_MSG((is_callable<f(X*, float)>::value == true), "mem-fun-ptr/ptr");
    HPX_TEST_MSG((is_callable<f(X const*, float)>::value == false), "mem-fun-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<f(X&, float)>::value == true), "mem-fun-ptr/lvref");
    HPX_TEST_MSG((is_callable<f(X const&, float)>::value == false), "mem-fun-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(X), float)>::value == true), "mem-fun-ptr/rvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(X const), float)>::value == false), "mem-fun-ptr/const-rvref");

    typedef int (X::*fc)(double) const;
    HPX_TEST_MSG((is_callable<fc(X*, float)>::value == true), "const-mem-fun-ptr/ptr");
    HPX_TEST_MSG((is_callable<fc(X const*, float)>::value == true), "const-mem-fun-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<fc(X&, float)>::value == true), "const-mem-fun-ptr/lvref");
    HPX_TEST_MSG((is_callable<fc(X const&, float)>::value == true), "const-mem-fun-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(X), float)>::value == true), "const-mem-fun-ptr/rvref");
    HPX_TEST_MSG((is_callable<fc(BOOST_RV_REF(X const), float)>::value == true), "const-mem-fun-ptr/const-rvref");
}

void member_object_pointers()
{
    using hpx::traits::is_callable;

    typedef int (X::*f);
    HPX_TEST_MSG((is_callable<f(X*)>::value == true), "mem-obj-ptr/ptr");
    HPX_TEST_MSG((is_callable<f(X const*)>::value == true), "mem-obj-ptr/const-ptr");
    HPX_TEST_MSG((is_callable<f(X&)>::value == true), "mem-obj-ptr/lvref");
    HPX_TEST_MSG((is_callable<f(X const&)>::value == true), "mem-obj-ptr/const-lvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(X))>::value == true), "mem-obj-ptr/rvref");
    HPX_TEST_MSG((is_callable<f(BOOST_RV_REF(X const))>::value == true), "mem-obj-ptr/const-rvref");
}

void function_objects()
{
    using hpx::traits::is_callable;

    HPX_TEST_MSG((is_callable<X(int)>::value == true), "fun-obj/value");
    HPX_TEST_MSG((is_callable<X const(int)>::value == false), "fun-obj/const-value");
    HPX_TEST_MSG((is_callable<X*(int)>::value == false), "fun-obj/ptr");
    HPX_TEST_MSG((is_callable<X const*(int)>::value == false), "fun-obj/const-ptr");
    HPX_TEST_MSG((is_callable<X&(int)>::value == true), "fun-obj/lvref");
    HPX_TEST_MSG((is_callable<X const&(int)>::value == false), "fun-obj/const-lvref");
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    HPX_TEST_MSG((is_callable<X&&(int)>::value == true), "fun-obj/rvref");
    HPX_TEST_MSG((is_callable<X const&&(int)>::value == false), "fun-obj/const-rvref");
#   endif

    HPX_TEST_MSG((is_callable<Xc(int)>::value == true), "const-fun-obj/value");
    HPX_TEST_MSG((is_callable<Xc const(int)>::value == true), "const-fun-obj/const-value");
    HPX_TEST_MSG((is_callable<Xc*(int)>::value == false), "const-fun-obj/ptr");
    HPX_TEST_MSG((is_callable<Xc const*(int)>::value == false), "const-fun-obj/const-ptr");
    HPX_TEST_MSG((is_callable<Xc&(int)>::value == true), "const-fun-obj/lvref");
    HPX_TEST_MSG((is_callable<Xc const&(int)>::value == true), "const-fun-obj/const-lvref");
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    HPX_TEST_MSG((is_callable<Xc&&(int)>::value == true), "const-fun-obj/rvref");
    HPX_TEST_MSG((is_callable<Xc const&&(int)>::value == true), "const-fun-obj/const-rvref");
#   endif
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
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

    return hpx::util::report_errors();
}
