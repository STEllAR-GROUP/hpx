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

///////////////////////////////////////////////////////////////////////////////
// Define a base component which exposes the required interface
struct A : hpx::components::abstract_managed_component_base<A>
{
    A() { hpx::cout << "A::A\n" << hpx::flush; }
    virtual ~A() { hpx::cout << "A::~A\n" << hpx::flush; }

    virtual void print() const = 0;

    // It is not possible to bind a virtual function to an action, thus we
    // bind a simple forwarding function, which is not virtual.
    void print_nonvirt() const { print(); }
    HPX_DEFINE_COMPONENT_ACTION(A, print_nonvirt, print_action);
};

HPX_DEFINE_GET_COMPONENT_TYPE(A);

typedef A::print_action print_action;
HPX_REGISTER_ACTION_DECLARATION(print_action);
HPX_REGISTER_ACTION(print_action);

///////////////////////////////////////////////////////////////////////////////
// Define a component which implements the required interface by deriving from
// the base component 'A' defined above.
struct B : A, hpx::components::managed_component_base<B>
{
    typedef B type_holder;
    typedef A base_type_holder;

    B() { hpx::cout << "B::B\n" << hpx::flush; }
    ~B() { hpx::cout << "B::~B\n" << hpx::flush; }

    void print() const { hpx::cout << "B::print\n" << hpx::flush; }
};

typedef hpx::components::managed_component<B> server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(server_type, B, "A");

///////////////////////////////////////////////////////////////////////////////
// Define a client side representation for a remote component instance 'B',
// Note: this client has no notion of using 'B', but wraps the base 'A' only.
struct client : hpx::components::client_base<client, A>
{
    typedef hpx::components::client_base<client, A> base_type;

    client(hpx::shared_future<hpx::id_type> const& gid)
      : base_type(gid)
    {}

    void print()
    {
        print_action act;
        act(base_type::get_id());
    }
};

///////////////////////////////////////////////////////////////////////////////
int main()
{
    // Use the client class to invoke the print functionality of the derived
    // component 'B'.
    client hw(hpx::components::new_<B>(hpx::find_here()));
    hw.print();

    return 0;
}

