//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERVER_MANAGED_CENTRAL_TUPLESPACE_MAR_29_2013_0237PM)
#define HPX_SERVER_MANAGED_CENTRAL_TUPLESPACE_MAR_29_2013_0237PM

#include <boost/unordered_map.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/util/storage/tuple.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/local/mutex.hpp>

#define TS_DEBUG

///////////////////////////////////////////////////////////////////////////////
namespace examples { namespace server
{

    ///////////////////////////////////////////////////////////////////////////
    /// This class is a simple central tuplespace (MCTS) as an HPX component. An HPX
    /// component is a class that:
    ///
    ///     * Inherits from a component base class (either
    ///       \a hpx::components::managed_component_base or
    ///       \a hpx::components::simple_component_base).
    ///     * Exposes methods that can be called asynchronously and/or remotely.
    ///       These constructs are known as HPX actions.
    ///
    /// By deriving this component from \a locking_hook the runtime system 
    /// ensures that all action invocations are serialized. That means that 
    /// the system ensures that no two actions are invoked at the same time on
    /// a given component instance. This makes the component thread safe and no
    /// additional locking has to be implemented by the user.
    ///
    /// Components are first-class objects in HPX. This means that they are
    /// globally addressable; all components have a unique GID.
    ///
    /// The MCTS will store all tuples from any objects in a central locality,
    /// to demonstrate the basic function
    ///
    /// (from JavaSpace)
    /// write,
    /// read,
    /// take 
    ///
    /// each has the last argument as a timeout value, pre-defined WAIT_FOREVER, NO_WAIT
    /// users can also provide its own timeout values.
    /// 
    /// uses mutex, will hurt performance.
    ///
    //[simple_central_tuplespace_server_inherit
    class simple_central_tuplespace
      : public hpx::components::locking_hook<
            hpx::components::simple_component_base<simple_central_tuplespace> 
        >
    //]
    {
        public:

            typedef hpx::util::storage::tuple tuple_type;
            typedef hpx::util::storage::tuple::elem_type elem_type;
            typedef hpx::util::storage::tuple::key_type key_type;
            typedef hpx::lcos::local::mutex mutex_type;
            
            typedef boost::unordered_multimap<key_type, tuple_type> tuples_type;
            typedef tuples_type::iterator tuples_iterator_type;

            static char * mutex_desc;

            // pre-defined timeout values
            enum {
                WAIT_FOREVER = -1, // <0 means blocking
                NO_WAIT = 0
            };

            //[simple_central_tuplespace_server_ctor
            simple_central_tuplespace() : mutex_(mutex_desc) {}
            //]

            ///////////////////////////////////////////////////////////////////////
            // Exposed functionality of this component.

            //[simple_accumulator_methods

            // put tuple into tuplespace
            // out function
            int write(tuple_type tuple)
            {
                if(tuple.empty())
                {
                    return -1;
                }

                tuple_type::iterator it = tuple.begin();
                key_type key = hpx::util::any_cast<key_type>(*it);

#if defined(TS_DEBUG)
                std::cerr<< " trying MUTEX_LOCK: in write.\n";
#endif

                mutex_.lock();
                tuples_.insert(std::pair<key_type, tuple_type>(key, tuple));

#if defined(TS_DEBUG)
                std::cerr<< " trying MUTEX_UNLOCK: in write.\n";
#endif
                mutex_.unlock();

                return 0;
            }

            // read from tuplespace
            // rd function
            // non-const b/c will change mutex_ value
            tuple_type read(const key_type& key, long timeout) 
            {
                tuple_type result;
                hpx::util::high_resolution_timer t;

                if(tuples_.empty())
                {
                    return result;
                }

                do
                {
#if defined(TS_DEBUG)
                std::cerr<< " trying MUTEX_LOCK: in read.\n";
#endif
                    mutex_.lock();

                    if(tuples_.empty())
                    {
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in read (empty tuples_).\n";
#endif
                        mutex_.unlock();
                        continue; 
                    }


                    tuples_iterator_type it = tuples_.find(key);
                    if(it == tuples_.end())
                    {
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in read (unfound).\n";
#endif
                        mutex_.unlock();
                        continue; // not found
                    }

                    result = it->second;
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in read (found).\n";
#endif
                    mutex_.unlock();

                    break; // found
                } while((timeout < 0) || (timeout > t.elapsed()));

                return result; 
            }

            // take from tuplespace
            // in function
            tuple_type take(const key_type& key, long timeout)
            {
                tuple_type result;
                hpx::util::high_resolution_timer t;

                if(tuples_.empty())
                {
                    return result;
                }

                do
                {
#if defined(TS_DEBUG)
                    std::cerr<< " trying MUTEX_LOCK: in take.\n";
#endif
                    mutex_.lock();

                    if(tuples_.empty())
                    {
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in take (empty tuples).\n";
#endif
                        mutex_.unlock();
                        continue; 
                    }

                    tuples_iterator_type it = tuples_.find(key);
                    if(it == tuples_.end())
                    {
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in take (unfound).\n";
#endif
                        mutex_.unlock();
                        continue; // not found
                    }

                    result = it->second;
                    tuples_.erase(it);
#if defined(TS_DEBUG)
                        std::cerr<< " trying MUTEX_UNLOCK: in take (found).\n";
#endif
                    mutex_.unlock();

                    break; // found
                } while((timeout < 0) || (timeout > t.elapsed()));

                return result; 
            }

            //]



            ///////////////////////////////////////////////////////////////////////
            // Each of the exposed functions needs to be encapsulated into an
            // action type, generating all required boilerplate code for threads,
            // serialization, etc.

            //[simple_central_tuplespace_action_types
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, write);
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, read);
            HPX_DEFINE_COMPONENT_ACTION(simple_central_tuplespace, take);
            //]

            //[simple_central_tuplespace_server_data_member
        private:
            tuples_type tuples_;
            mutex_type mutex_;
            //]
    };
}} // examples::server

char * examples::server::simple_central_tuplespace::mutex_desc = (char*)"simple_central_tuplespace_mutex";

//[simple_central_tuplespace_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::write_action,
    simple_central_tuplespace_write_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::read_action,
    simple_central_tuplespace_read_action);

HPX_REGISTER_ACTION_DECLARATION(
    examples::server::simple_central_tuplespace::take_action,
    simple_central_tuplespace_take_action);
//]

#undef TS_DEBUG

#endif

