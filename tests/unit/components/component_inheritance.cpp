////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>

bool a_ctor = false;
bool a_dtor = false;
bool b_ctor = false;
bool b_dtor = false;
bool c_ctor = false;
bool c_dtor = false;
bool d_ctor = false;
bool d_dtor = false;
bool e_ctor = false;
bool e_dtor = false;

///////////////////////////////////////////////////////////////////////////////
// Abstract
struct A : hpx::components::abstract_managed_component_base<A>
{
    A() { a_ctor = true; }
    virtual ~A() { a_dtor = true; }

    virtual std::string test0() const = 0;
    std::string test0_nonvirt() const { return test0(); }
    HPX_DEFINE_COMPONENT_CONST_ACTION(A, test0_nonvirt, test0_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(A);

typedef A::test0_action test0_action;
HPX_REGISTER_ACTION_DECLARATION(test0_action);
HPX_REGISTER_ACTION(test0_action);

///////////////////////////////////////////////////////////////////////////////
// Concrete
struct B : A, hpx::components::managed_component_base<B>
{
    typedef B type_holder;
    typedef A base_type_holder;

    B() { b_ctor = true; }
    ~B() { b_dtor = true; }

    virtual std::string test0() const { return "B"; }

    virtual std::string test1() const { return "B"; }
    std::string test1_nonvirt() const { return test1(); }
    HPX_DEFINE_COMPONENT_CONST_ACTION(A, test1_nonvirt, test1_action);
};

typedef hpx::components::managed_component<B> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, B, "A");

typedef B::test1_action test1_action;
HPX_REGISTER_ACTION_DECLARATION(test1_action);
HPX_REGISTER_ACTION(test1_action);

///////////////////////////////////////////////////////////////////////////////
// Abstract
struct C : A, hpx::components::abstract_managed_component_base<C>
{
    typedef C type_holder;
    typedef A base_type_holder;

    C() { c_ctor = true; }
    ~C() { c_dtor = true; }

    virtual std::string test0() const { return "C"; }

    virtual std::string test2() const = 0;
    std::string test2_nonvirt() const { return test1(); }
    HPX_DEFINE_COMPONENT_CONST_ACTION(A, test1_nonvirt, test1_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(C);

typedef C::test2_action test2_action;
HPX_REGISTER_ACTION_DECLARATION(test2_action);
HPX_REGISTER_ACTION(test2_action);

///////////////////////////////////////////////////////////////////////////////
// Concrete
struct D : B, hpx::components::managed_component_base<D>
{
    typedef D type_holder;
    typedef B base_type_holder;

    D() { d_ctor = true; }
    ~D() { d_dtor = true; }

    std::string test0() const { return "D"; }

    std::string test1() const { return "D"; }
};

typedef hpx::components::managed_component<D> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, D, "B");

///////////////////////////////////////////////////////////////////////////////
// Concrete
struct E : C, hpx::components::managed_component_base<E>
{
    typedef E type_holder;
    typedef C base_type_holder;

    D() { e_ctor = true; }
    ~D() { e_dtor = true; }

    std::string test0() const { return "E"; }

    std::string test2() const { return "E"; }
};

typedef hpx::components::managed_component<E> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, E, "C");

///////////////////////////////////////////////////////////////////////////////
struct clientA : hpx::components::clientA_base<clientA, A>
{
    typedef hpx::components::clientA_base<clientA, A> base_type;

    clientA(hpx::future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_gid());
    }
};

///////////////////////////////////////////////////////////////////////////////
struct clientB : hpx::components::clientB_base<clientB, B>
{
    typedef hpx::components::clientB_base<clientB, B> base_type;

    clientB(hpx::future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_gid());
    }

    std::string test1()
    {
        test1_action act;
        return act(base_type::get_gid());
    }
};

///////////////////////////////////////////////////////////////////////////////
struct clientC : hpx::components::clientB_base<clientC, C>
{
    typedef hpx::components::clientB_base<clientC, C> base_type;

    clientC(hpx::future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_gid());
    }

    std::string test2()
    {
        test2_action act;
        return act(base_type::get_gid());
    }
};

///////////////////////////////////////////////////////////////////////////////
void reset_globals()
{
    a_ctor = false;
    a_dtor = false;
    b_ctor = false;
    b_dtor = false;
    c_ctor = false;
    c_dtor = false;
    d_ctor = false;
    d_dtor = false;
    e_ctor = false;
    e_dtor = false;
}

int main()
{
    ///////////////////////////////////////////////////////////////////////////

    { // Client to A, instance of B
        clientA obj(hpx::components::new_<B>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "B");
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(!c_ctor); HPX_TEST(!c_dtor);
        HPX_TEST(!d_ctor); HPX_TEST(!d_dtor);
        HPX_TEST(!e_ctor); HPX_TEST(!e_dtor);
    }

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to B, instance of B
        clientB obj(hpx::components::new_<B>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "B");
        HPX_TEST_EQ(obj.test1(), "B");
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(!c_ctor); HPX_TEST(!c_dtor);
        HPX_TEST(!d_ctor); HPX_TEST(!d_dtor);
        HPX_TEST(!e_ctor); HPX_TEST(!e_dtor);
    }

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to B, instance of D
        clientB obj(hpx::components::new_<D>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "D"); 
        HPX_TEST_EQ(obj.test1(), "D"); 
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(!c_ctor); HPX_TEST(!c_dtor);
        HPX_TEST(d_ctor); HPX_TEST(d_dtor);
        HPX_TEST(!e_ctor); HPX_TEST(!e_dtor);
    }

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to A, instance of D
        clientA obj(hpx::components::new_<D>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "D"); 
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(!c_ctor); HPX_TEST(!c_dtor);
        HPX_TEST(d_ctor); HPX_TEST(d_dtor);
        HPX_TEST(!e_ctor); HPX_TEST(!e_dtor);
    }

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to C, instance of E
        clientC obj(hpx::components::new_<E>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "E"); 
        HPX_TEST_EQ(obj.test2(), "E"); 
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(c_ctor); HPX_TEST(c_dtor);
        HPX_TEST(!d_ctor); HPX_TEST(!d_dtor);
        HPX_TEST(e_ctor); HPX_TEST(e_dtor);
    }

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to A, instance of E
        clientA obj(hpx::components::new_<E>(hpx::find_here()));
    
        HPX_TEST_EQ(obj.test0(), "E"); 
    
        HPX_TEST(a_ctor); HPX_TEST(a_dtor);
        HPX_TEST(b_ctor); HPX_TEST(b_dtor);
        HPX_TEST(c_ctor); HPX_TEST(c_dtor);
        HPX_TEST(!d_ctor); HPX_TEST(!d_dtor);
        HPX_TEST(e_ctor); HPX_TEST(e_dtor);
    }

    reset_globals();

    return 0;
}

