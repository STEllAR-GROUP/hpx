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

// FIXME: This facility hopefully will exist in HPX sometime in the future 
// (in a form that can take up to HPX_ACTION_ARGUMENT_LIMIT ctor args).
template <typename Component>
hpx::id_type new_(hpx::id_type const& locality)
{
    return hpx::components::stub_base<Component>::create(locality);
}

struct A : hpx::components::abstract_managed_component_base<A> 
{
    A() { hpx::cout << "A::A\n" << hpx::flush; }
    virtual ~A() { hpx::cout << "A::~A\n" << hpx::flush; }

    virtual void print() const = 0;

    void print_nonvirt() const { print(); }

    HPX_DEFINE_COMPONENT_CONST_ACTION(A, print_nonvirt, print_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(A);

typedef A::print_action print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

struct client
  : hpx::components::client_base<client, hpx::components::stub_base<A> >
{
    typedef hpx::components::client_base<client, hpx::components::stub_base<A> >
        base_type;

    client() {}
    client(hpx::id_type const& gid) : base_type(gid) {} 

    virtual void print() { hpx::async<print_action>(this->gid_).get(); } 
};

///////////////////////////////////////////////////////////////////////////////

struct B : A, hpx::components::managed_component_base<B>
{
    typedef B type_holder;
    typedef A base_type_holder;

    B() { hpx::cout << "B::B\n" << hpx::flush; }
    virtual ~B() { hpx::cout << "B::~B\n" << hpx::flush; }

    void print() const { hpx::cout << "B::print\n" << hpx::flush; } 
};

typedef hpx::components::managed_component<B> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, B, "A");

///////////////////////////////////////////////////////////////////////////////
int main()
{
    client hw(new_<B>(hpx::find_here()));

    hw.print();    

    return 0; 
}

