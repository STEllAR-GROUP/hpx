//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/lcos/async.hpp>

#include <tests/unit/agas/components/simple_refcnt_checker.hpp>
#include <tests/unit/agas/components/managed_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::naming::id_type;
using hpx::naming::get_management_type_name;
using hpx::naming::get_locality_id_from_id;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::actions::plain_action3;

using hpx::lcos::async;

using hpx::test::simple_object;
using hpx::test::managed_object;

using hpx::util::report_errors;

using hpx::cout;
using hpx::flush;
using hpx::find_here;

void split(
    id_type const& from
  , id_type const& target
  , boost::uint16_t old_credit
    );

typedef plain_action3<
    // arguments
    id_type const&
  , id_type const&
  , boost::uint16_t
    // function
  , split
> split_action;

HPX_REGISTER_PLAIN_ACTION(split_action);

void split(
    id_type const& from
  , id_type const& target
  , boost::uint16_t old_credit
    )
{
    std::cout << "[" << find_here() << "/" << target << "]: " << old_credit << ", " << target.get_credit() << "\n";

    // If we have more credits than the sender, then we're done. 
    if (old_credit < target.get_credit())
        return; 

    id_type const here = find_here();

    if (get_locality_id_from_id(from) == get_locality_id_from_id(here)) 
        throw std::logic_error("infinite recursion detected, split was "
                               "invoked locally");

    // Recursively call split on the sender locality.
    async<split_action>(from, here, target, target.get_credit()).get();
}

///////////////////////////////////////////////////////////////////////////////
template <
    typename Client
>
void hpx_test_main(
    variables_map& vm
    )
{
    {
        std::vector<id_type> remote_localities;

        typedef typename Client::server_type server_type;

        component_type ctype = get_component_type<server_type>();

        if (!get_applier().get_remote_prefixes(remote_localities, ctype))
            throw std::logic_error("this test cannot be run on one locality");

        id_type const here = find_here();

        Client object(here);

        cout << "id: " << object.get_gid() << " "
             << get_management_type_name
                    (object.get_gid().get_management_type()) << "\n"
             << flush;

        async<split_action>(remote_localities[0]
                          , here
                          , object.get_gid()
                          , object.get_gid().get_credit()).get();
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        cout << std::string(80, '#') << "\n"
             << "simple component test\n"
             << std::string(80, '#') << "\n" << flush;

        hpx_test_main<simple_object>(vm);

        cout << std::string(80, '#') << "\n"
             << "managed component test\n"
             << std::string(80, '#') << "\n" << flush;

        hpx_test_main<managed_object>(vm);
    }

    finalize();
    return report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "delay"
        , value<boost::uint64_t>()->default_value(1000)
        , "number of milliseconds to wait for object destruction")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

