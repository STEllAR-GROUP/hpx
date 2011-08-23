// Copyright (c) 2011 John West <john.e.west@gmail.com>
// Copyright (c) 2011 Matt Anderson <matt@phys.lsu.edu>
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This example code is a candidate for the "next simplest possible" HPX program that is in some way
// instructive. As such the code in it may not represent the best way of getting something done;
// it is primarily meant to familiarize the reader with the essential steps to getting an HPX 
// code up and running.
//
// The code starts out in the main routine, but the only function of the main() in this example is to 
// (using the Boost helper routines) package up the command line options and call hpx_init(). 
// hpx_init() initializes the runtime environment and then calls hpx_main(). This routine must be present, 
// by name, in any HPX program. (For more about Boost, see http://www.boost.org/ .) 
//
// hpx_main() does some bookkeeping and setup and then causes parallel work to be done by invoking
// the futures registered earlier in the program.
// 
// To run the program
//       ./basic_example -r -t 2 -a localhost:5005 -x localhost:5006. 
//       - The "-t 2" option runs this program on two OS threads (one for each future). 
//       - The "-r" option runs the AGAS server (necessary if it isn't already running somewhere else).
//       - The "-a localhost:5005" option places the AGAS server on your machine and specifies that it
//         use port 5005 (this example assumes that ports 5005 and 5006 are open on your system. 
//         If they aren't, substitute two open port numbers appropriate for you).
//       - The "-x localhost:5006" specifies that HPX should include "localhost" in the runtime and use
//         port 5006.
// 
// If you'd like to get some validation that getnumber_action is being run on two different threads, you 
// can ask HPX to output a logfile via an environment variable. If you are using bash type
// 
//    export HPX_LOGLEVEL=6
// 
// for very detailed information or
// 
//    export HPX_LOGLEVEL=2
// 
// for more high level information.
// 
// After you execute the application HPX creates a log file of the form hpx.PID.log. You can search this file
// for the string getnumber_action and see that the thread IDs differ.

//C++ include to permit output to the screen
#include <iostream>

//HPX includes
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

//Boost includes
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
    // Configure application-specific options
    po::options_description desc_commandline ("usage:basic_example");

    /* 
       Initialize and run HPX; there are (at least) two ways to do this. One is using hpx_init
       as shown below -- this is analogous to mpi_init and creates the hpx runtime and performs
       other setup as directed by command line arguments. The other method is explicitly via
       a series of calls to create the runtime manually with the desired characteristics. The
       two methods have equivalent results (ie., and hpx runtime is created). 
    */
    int retcode = hpx::init(desc_commandline, argc, argv);
    return retcode;

}

/* 
    These three delcarations are here, before hpx_main, because they will be needed by that
    block of code. As you will see when reading below, these declarations setup the routines
    that will do parallel work in this example application and register them with the HPX
    runtime.
*/

    /*
    getnumber() is the routine in which the "work" for this example is done. In this case the 
    work is trivial -- the integer 6 is returned from each invocation -- but in a real code
    there would be a calculation in here representing some piece of capability in the underlying
    application. In fact, there would likely be (many) more than one such routine for each of the
    units of work that need to be done. We need the declaration here because of the code that 
    follows it.
    */
int getnumber();
naming::id_type getnumber2();

    /*
    This typedef takes the work routine in this example, getnumber(), and uses it to create a 
    new type. HPX will use this declaration in the next step when it registers the routine
    with the runtime.
    
    In this case the declaration is a plain_result_action0, which means that the function
    being registered as an action takes no (0) arguments but provides a result (an integer).
    The return type, int, is specified in the template, as is the name of the function that
    is performing this action's work. There are other variants of this action, for example
    that take 2 arguments (actions::plain_result_action<(return_type),arg1,arg2,function>), 
    variants that return no result, and so on. Until the documentation improves look in 
    hpx/runtime/actions/plain_action.hpp and related files for the form of these variants.
    */
//typedef 
//    actions::plain_result_action0<int, getnumber> 
//getnumber_action;

typedef 
    actions::plain_result_action0<naming::id_type, getnumber2> 
getnumber_action;

    /*
    This step registers the action just delcared, getnumber_action, based on the routine
    getnumber() with the HPX runtime. Because this call is made by all localities running 
    this example code, then when a block of code queries the runtime to see who is available
    to provide computations on behalf of getnumber_action all localities will report back.
    */
HPX_REGISTER_PLAIN_ACTION(getnumber_action);

/* 
    hpx_main must exist with this name in any hpx program. In this example it is called
    by hpx_init(), but this routine is also in the call chain when you choose to create the 
    hpx runtime explicitly as well.
*/
struct data
{
    data()
      : val_(0)
    {}
    ~data() {}

    int val_;
    std::vector<int> x_;
    bool proceed_;

private:
    // serialization support
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & val_ & x_ & proceed_;
    }
};


int hpx_main(po::variables_map &vm)
{
    int result = 0;
    double elapsed = 0.0;

    /* Get the runtime application configuration information from the runtime created implicitly
       by hpx_init() */
    //runtime& rt = get_runtime();

    /* 
        In this block of code we are querying all localities that the current runtime knows
        about to see which of them have registered to provide the action we need, ie. the 
        getnumber_action. This list represents the universe of computational resources that we 
        can schedule tasks for completion on within this runtime.
    */

        std::vector<naming::id_type> prefixes;
        applier::applier& appl = applier::get_applier();

        /* This call provides the global identifier (gid) for the locality running this block
           of code. Ie., it provides a handle to "this" processor. */
        naming::id_type this_prefix = appl.get_runtime_support_gid();
    
        /* 
            This code checks whether the action we want, in this case getnumber_action, is
            registered in other localities. If it is registered in at least one other locality
            then we store the global identifier of the first of those in the list into the variable
            that_prefix, which we'll use later.
        */
    
        /* This declaration creates a component type variable that we use to search the registry of
           actions in the code below. */
        components::component_type type = 
            components::get_component_type<components::server::plain_function<getnumber_action> >();
        // Declaration used to store the first gid (if any) of the remote prefixes
        naming::id_type that_prefix;
    
        // Search the remote localities (ie., localities that aren't this locality) for those that
        // have registered to provide the getnumber_action service.
        if (appl.get_remote_prefixes(prefixes, type)) {
            // If there is at least one such remote locality, store the first gid in the list
            that_prefix = prefixes[0];
        }
        else {
            // There are no remote localities that provide the getnumber_action we want, so set the 
            // remote prefix variable to the gid of our local locality (executes the function locally only)
            that_prefix = this_prefix;
        }

    
    {
        // Create a timer so see how its done
        util::high_resolution_timer t;

        /*
            Delcare two Local Control Objects (lcos) as futures that perform the getnumber_action we
            registered earlier. Since we have not declared a thread scheduler in this application we
            have to tell the runtime where to run the requested action. Having to tell the runtime 
            where to run the request is obviously not the best use of the ParalleX model. You would
            usually want to give the runtime the freedom to put the work in the place where it can
            complete the fastest, and this may vary from run to run. For simplicity of this most basic
            example, however, we are telling the LCO where to run the action. Note that if our action 
            required parameters, they would appear in the list after "that_prefix".
        */
        lcos::eager_future<getnumber_action> n1(that_prefix);
        lcos::eager_future<getnumber_action> n2(this_prefix);

        //insert busy work
        
        /*   Calling the "get" method on a future causes the application to return the value that was
            previouly computed (if the runtime was able to overlap some of the computation) or to 
            block until the requested value is available. Calling "get" tells the runtime that 
            you've got to have that value to continue with your computation and there is no other
            work to be done.
        */

        // Access memory
        components::access_memory_block<data> val1( components::stubs::memory_block::get(n1.get()) );
        components::access_memory_block<data> val2( components::stubs::memory_block::get(n2.get()) );
        std::cout << " Result " << val1->val_ << " " << val2->val_ << std::endl;
        std::cout << " Vector Result " << val1->x_[0] << " " << val2->x_[0] << std::endl;
        std::cout << " Proceed Result " << val1->proceed_ << " " << val2->proceed_ << std::endl;

        //std::cout << " Result: " << n1.get() << " " << n2.get() << std::endl;
        //result = n1.get()+n2.get();

        // What is the elapsed time?
        elapsed = t.elapsed();

        // Print out a completion message. The correct answer is 12 for this example.
        std::cout << "Achieved result of " << result << " in " << elapsed << " seconds."<< std::endl;
    }

    // Initiate shutdown of the runtime systems on all localities
    hpx::finalize();
    return 0;
}

hpx::actions::manage_object_action<data> const manage_data =
        hpx::actions::manage_object_action<data>();

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::actions::manage_object_action<data>, manage_object_action_data)

// The routine that does the "work" for this example.
int getnumber ()
{  
    int a,b;

    a = 2;
    b = 3;

    return (a*b);
    //return result;
}

naming::id_type getnumber2 ()
{  
    naming::id_type here = applier::get_applier().get_runtime_support_gid();
    naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(data), manage_data);

    components::access_memory_block<data> val(
                components::stubs::memory_block::checkout(result));

    int locality = get_prefix_from_id( here );

    if ( locality == 1 ) {
      val->val_ = 6;
      val->x_.push_back(1);
      val->x_.push_back(2);
      val->x_.push_back(3);
      val->proceed_ = true;
    } else {
      val->val_ = 3;
      val->x_.push_back(4);
      val->x_.push_back(5);
      val->x_.push_back(6);
      val->proceed_ = false;
    }

    return result;
}

