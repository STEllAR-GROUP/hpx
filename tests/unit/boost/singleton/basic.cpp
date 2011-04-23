/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#ifndef SINGLETON_TYPE
#   define SINGLETON_TYPE singleton
#   include <boost/utility/singleton.hpp>
#endif

#include <hpx/util/lightweight_test.hpp>
#include <boost/bind.hpp>

template< int Id >
struct X : public boost:: SINGLETON_TYPE < X<Id> >
{
    unsigned val_ctor_called;
    unsigned val_instance;

    explicit X(boost::restricted)
        : val_ctor_called(0xbeef) 
    {
        val_instance = ++n_instances; 
    }

    ~X()
    {
        HPX_TEST(val_instance == n_instances);
        --n_instances;
    }

    unsigned f() 
    {
        HPX_TEST(val_ctor_called == 0xbeef);
        return 0xbeef;
    }

    unsigned fc()
    {
        HPX_TEST(val_ctor_called == 0xbeef);
        return 0xbeef;
    }

    static unsigned n_instances; 
};

template< int Id >
unsigned X<Id>::n_instances = 0;


int main()
{
    {
        // test initialization through instance proxy's operator->
        HPX_TEST(! X<0>::n_instances);
        X<0>::instance->f();
        HPX_TEST(X<0>::n_instances == 1);
        X<0>::instance->f();
        HPX_TEST(X<0>::n_instances == 1);
    }
    {
        // test initialization through operator-> of lease
        HPX_TEST(! X<1>::n_instances);
        X<1>::lease lease;
        HPX_TEST(lease->f() == 0xbeef);
        HPX_TEST(X<1>::n_instances == 1);
        HPX_TEST(lease->f() == 0xbeef);
        HPX_TEST(X<1>::n_instances == 1);
    }
    {
        // test multiple, overlapping leases 
        HPX_TEST(! X<2>::n_instances);
        X<2>::lease lease1;
        HPX_TEST(lease1->f() == 0xbeef);
        HPX_TEST(lease1->f() == 0xbeef);
        HPX_TEST(X<2>::n_instances == 1);
        X<2>::lease lease2;
        HPX_TEST(X<2>::n_instances == 1);
        HPX_TEST(lease2->f() == 0xbeef);
        HPX_TEST(lease2->f() == 0xbeef);
        HPX_TEST(lease1->f() == 0xbeef);
        HPX_TEST(X<2>::n_instances == 1);
    } 
    {
        // test direct application of instance proxy's operator->*
        HPX_TEST(! X<3>::n_instances);
        HPX_TEST((X<3>::instance->*(& X<3>::f))() == 0xbeef);
        HPX_TEST((X<3>::instance->*(& X<3>::fc))() == 0xbeef);
        HPX_TEST(X<3>::instance->*(& X<3>::val_ctor_called) == 0xbeef);
        HPX_TEST(X<3>::n_instances == 1);
    }
    {
        // test direct application of lease's operator->*
        HPX_TEST(! X<4>::n_instances);
        X<4>::lease l;
        HPX_TEST((l->*(& X<4>::f))() == 0xbeef);
        HPX_TEST((l->*(& X<4>::fc))() == 0xbeef);
        HPX_TEST(l->*(& X<4>::val_ctor_called) == 0xbeef);
        HPX_TEST(X<4>::n_instances == 1);
    }
    {
        // test boost::bind with reference to instance proxy
        HPX_TEST(! X<5>::n_instances);
        HPX_TEST(boost::bind(& X<5>::f, ref(X<5>::instance))() == 0xbeef);
        HPX_TEST(boost::bind(& X<5>::fc, ref(X<5>::instance))() == 0xbeef);
        HPX_TEST(boost::bind(& X<5>::val_ctor_called, ref(X<5>::instance))() == 0xbeef);
        HPX_TEST(X<5>::n_instances == 1);
    }
    {
        // test boost::bind with lease object
        HPX_TEST(! X<6>::n_instances);
        HPX_TEST(boost::bind(& X<6>::f, X<6>::lease())() == 0xbeef);
        HPX_TEST(boost::bind(& X<6>::fc, X<6>::lease())() == 0xbeef);
        HPX_TEST(boost::bind(& X<6>::val_ctor_called, X<6>::lease())() == 0xbeef);
        HPX_TEST(X<6>::n_instances == 1);
    }

    // test destructors, that is before main() exits
    boost::destroy_singletons();

    return hpx::util::report_errors();
}


