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
#include <hpx/util/lightweight_test.hpp>

#include <string>

bool a_ctor = false;
bool a_dtor = false;
bool b_ctor = false;
bool b_dtor = false;
bool c_ctor = false;
bool c_dtor = false;

///////////////////////////////////////////////////////////////////////////////
// Abstract
struct A : hpx::components::abstract_managed_component_base<A>
{
    A() { a_ctor = true; }
    virtual ~A() { a_dtor = true; }

    virtual std::string test0() const = 0;
    std::string test0_nonvirt() const { return test0(); }
    HPX_DEFINE_COMPONENT_ACTION(A, test0_nonvirt, test0_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(A);

typedef A::test0_action test0_action;
HPX_REGISTER_ACTION_DECLARATION(test0_action);
HPX_REGISTER_ACTION(test0_action);

///////////////////////////////////////////////////////////////////////////////
// Abstract
struct B : A, hpx::components::abstract_managed_component_base<B>
{
    typedef hpx::components::abstract_managed_component_base<B>::wrapping_type
        wrapping_type;

    typedef B type_holder;
    typedef A base_type_holder;

    B() { b_ctor = true; }
    virtual ~B() { b_dtor = true; }

    virtual std::string test0() const { return "B"; }

    virtual std::string test1() const = 0;
    std::string test1_nonvirt() const { return test1(); }
    HPX_DEFINE_COMPONENT_ACTION(B, test1_nonvirt, test1_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(B);

typedef B::test1_action test1_action;
HPX_REGISTER_ACTION_DECLARATION(test1_action);
HPX_REGISTER_ACTION(test1_action);

///////////////////////////////////////////////////////////////////////////////
// Concrete
struct C : B, hpx::components::managed_component_base<C>
{
    typedef hpx::components::managed_component_base<C>::wrapping_type
        wrapping_type;

    typedef C type_holder;
    typedef B base_type_holder;

    C() { c_ctor = true; }
    ~C() { c_dtor = true; }

    std::string test0() const { return "C"; }

    std::string test1() const { return "C"; }

    std::string test2() const { return "C"; }
    HPX_DEFINE_COMPONENT_ACTION(C, test2, test2_action);
};

typedef hpx::components::managed_component<C> serverC_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(serverC_type, C, "B");

typedef C::test2_action test2_action;
HPX_REGISTER_ACTION_DECLARATION(test2_action);
HPX_REGISTER_ACTION(test2_action);

///////////////////////////////////////////////////////////////////////////////
struct clientA : hpx::components::client_base<clientA, A>
{
    typedef hpx::components::client_base<clientA, A> base_type;

    clientA(hpx::shared_future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
struct clientB : hpx::components::client_base<clientB, B>
{
    typedef hpx::components::client_base<clientB, B> base_type;

    clientB(hpx::shared_future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_id());
    }

    std::string test1()
    {
        test1_action act;
        return act(base_type::get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
struct clientC : hpx::components::client_base<clientC, C>
{
    typedef hpx::components::client_base<clientC, C> base_type;

    clientC(hpx::shared_future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    std::string test0()
    {
        test0_action act;
        return act(base_type::get_id());
    }

    std::string test1()
    {
        test1_action act;
        return act(base_type::get_id());
    }

    std::string test2()
    {
        test2_action act;
        return act(base_type::get_id());
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
}

int main()
{
    ///////////////////////////////////////////////////////////////////////////

    { // Client to A, instance of C
        clientA obj(hpx::components::new_<C>(hpx::find_here()));

        HPX_TEST_EQ(obj.test0(), "C");
    }

    HPX_TEST(a_ctor); HPX_TEST(a_dtor);
    HPX_TEST(b_ctor); HPX_TEST(b_dtor);
    HPX_TEST(c_ctor); HPX_TEST(c_dtor);

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to B, instance of C
        clientB obj(hpx::components::new_<C>(hpx::find_here()));

        HPX_TEST_EQ(obj.test0(), "C");
        HPX_TEST_EQ(obj.test1(), "C");
    }

    HPX_TEST(a_ctor); HPX_TEST(a_dtor);
    HPX_TEST(b_ctor); HPX_TEST(b_dtor);
    HPX_TEST(c_ctor); HPX_TEST(c_dtor);

    reset_globals();

    ///////////////////////////////////////////////////////////////////////////

    { // Client to C, instance of C
        clientC obj(hpx::components::new_<C>(hpx::find_here()));

        HPX_TEST_EQ(obj.test0(), "C");
        HPX_TEST_EQ(obj.test1(), "C");
        HPX_TEST_EQ(obj.test2(), "C");
    }

    HPX_TEST(a_ctor); HPX_TEST(a_dtor);
    HPX_TEST(b_ctor); HPX_TEST(b_dtor);
    HPX_TEST(c_ctor); HPX_TEST(c_dtor);

    reset_globals();

    return 0;
}

