////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_63C75B03_87A6_428E_99A7_F91027E0D463)
#define HPX_63C75B03_87A6_428E_99A7_F91027E0D463

#include <map>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/agas/traits.hpp>

namespace hpx { namespace components { namespace agas { namespace server
{

struct HPX_COMPONENT_EXPORT refcnt_service
  : simple_component_base<refcnt_service>
{
    typedef hpx::lcos::mutex mutex_type;
    typedef std::map<naming::gid_type, boost::uint64_t> registry_type;
    typedef registry_type::key_type key_type;
    typedef registry_type::mapped_type mapped_type; 
    typedef boost::fusion::vector2<boost::uint64_t, component_type>
        decrement_result_type;

    enum actions
    {
        refcnt_increment,
        refcnt_decrement
    };

  private:
    mutex_type mutex_;
    registry_type registry_;

  public:
    refcnt_service()
    { hpx::agas::traits::initialize_mutex(mutex_); }

    mapped_type
    increment(key_type const& key, mapped_type count)
    {
        try {
            mutex_type::scoped_lock l(mutex_);

            // FIXME: This copy is a one-off, but necessary because
            // strip_credit_from_gid() mutates. Fix this.
            naming::gid_type id = key;
            naming::strip_credit_from_gid(id);

            registry_type::iterator it = registry_.find(id);

            if (it == registry_.end())
            {
                std::pair<registry_type::iterator, bool> p = registry_.insert
                    (registry_type::value_type(id, HPX_INITIAL_GLOBALCREDIT));

                if (!p.second)
                    throw exception(out_of_memory);

                it = p.first;
            }

            return it->second += count;
        } catch (std::bad_alloc) {
            throw exception(out_of_memory);
        } catch (hpx::exception) {
            throw;
        } catch (...) {
            throw exception(internal_server_error);
        }
    }

    decrement_result_type 
    decrement(key_type const& key, mapped_type count)
    {
        using boost::fusion::at_c;

        try {
            mutex_type::scoped_lock l(mutex_);

            // FIXME: This copy is a one-off, but necessary because
            // strip_credit_from_gid() mutates. Fix this.
            naming::gid_type id = key;
            naming::strip_credit_from_gid(id);

            // TODO: This needs an exception message explaining why it's
            // invalid.
            if (count <= HPX_INITIAL_GLOBALCREDIT)
                throw exception(bad_parameter);

            decrement_result_type r(0, component_invalid);

            registry_type::iterator it = registry_.find(id);

            if (it != registry_.end())
            {
                if (it->second < count)
                    throw exception(bad_parameter,
                        "bogus credit found while decrementing global reference"
                        "count");

                // Remove the entry if it's been decremented to 0.
                if ((it->second -= count) == 0)
                    registry_.erase(it);
                else
                    at_c<0>(r) = it->second;
            }

            // Annoying-ish insert-on-decrement semantics reproduced to maintain
            // AGAS v1 behavior. 
            else if (count < HPX_INITIAL_GLOBALCREDIT)
            {
                std::pair<registry_type::iterator, bool> p = registry_.insert
                    (registry_type::value_type(id, HPX_INITIAL_GLOBALCREDIT));

                if (!p.second)
                    throw exception(out_of_memory);

                it = p.first;
               
                if (it->second < count)
                    throw exception(bad_parameter,
                        "bogus credit found while decrementing global reference"
                        "count");

                at_c<0>(r) = (it->second -= count);
            }
          
            // FIXME: This can't be implemented without access to the primary
            // namespace. 
            #if 0 
            if (at_c<0>(r) == 0)
            {
                if ((at_c<1>(r) = get_component_type(id)) == component_invalid)
                    throw exception(bad_component_type,
                        "unknown component type encountered while decrementing"
                        "global reference count to 0");
            }
            #endif
    
            return r; 

        } catch (std::bad_alloc) {
            throw exception(out_of_memory);
        } catch (hpx::exception) {
            throw;
        } catch (...) {
            throw exception(internal_server_error);
        }
    }

    typedef hpx::actions::result_action2<
        refcnt_service,
        mapped_type,                  // return type
        refcnt_increment,             // action type
        key_type const&, mapped_type, // arguments 
        &refcnt_service::increment
    > increment_action;

    typedef hpx::actions::result_action2<
        refcnt_service,
        decrement_result_type,        // return type
        refcnt_decrement,             // action type
        key_type const&, mapped_type, // arguments 
        &refcnt_service::decrement
    > decrement_action;
};

}}}}

#endif // HPX_63C75B03_87A6_428E_99A7_F91027E0D463

