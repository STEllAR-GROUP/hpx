////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_63C75B03_87A6_428E_99A7_F91027E0D463)
#define HPX_63C75B03_87A6_428E_99A7_F91027E0D463

#include <map>

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/mutex.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

namespace hpx { namespace components { namespace server
{

struct HPX_COMPONENT_EXPORT refcnt_service
  : simple_component_base<refcnt_service>
{
    typedef hpx::lcos::mutex mutex_type;
    typedef std::map<naming::gid_type, boost::uint64_t> registry_type;
    typedef registry_type::key_type key_type;
    typedef registry_type::mapped_type mapped_type; 

    enum actions
    {
        refcnt_increment,
        refcnt_decrement
    };

  private:
    mutex_type _mutex;
    registry_type _registry;

  public:
    refcnt()
    { hpx::agas::traits::initialize_mutex(_mutex); }

    boost::uint64_t
    increment(naming::gid_type const& key, boost::uint64_t count)
    {
        try {
            mutex_type::scoped_lock l(_mutex);

            // FIXME: This copy is a one-off, but necessary because
            // strip_credit_from_gid() mutates. Fix this.
            naming::gid_type id = key;
            naming::strip_credit_from_gid(id);

            registry_type::iterator it = _registry.find(id);

            if (it == _registry.end())
            {
                std::pair<registry_type::iterator, bool> p = _registry.insert
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

    boost::uint64_t
    decrement(naming::gid_type const& key, boost::uint64_t count)
    {
        try {
            mutex_type::scoped_lock l(_mutex);

            // FIXME: This copy is a one-off, but necessary because
            // strip_credit_from_gid() mutates. Fix this.
            naming::gid_type id = key;
            naming::strip_credit_from_gid(id);

            // TODO: This needs an exception message explaining why it's
            // invalid.
            if (count <= HPX_INITIAL_GLOBALCREDIT)
                throw exception(bad_parameter);

            registry_type::iterator it = _registry.find(id);

            if (it != _registry.end())
            {
                if (it->second < count)
                    throw exception(bad_parameter,
                        "bogus credit found while decrementing global reference"
                        "count");

                // Remove the entry if it's been decremented to 0.
                if ((it->second -= count) == 0)
                    _registry.erase(it);
                else
                    return it->second;
            }

            // Annoying-ish insert-on-decrement semantics reproduced to maintain
            // AGAS v1 behavior. 
            else if (count < HPX_INITIAL_GLOBALCREDIT)
            {
                std::pair<registry_type::iterator, bool> p = _registry.insert
                    (registry_type::value_type(id, HPX_INITIAL_GLOBALCREDIT));

                if (!p.second)
                    throw exception(out_of_memory);

                it = p.first;
               
                if (it->second < count)
                    throw exception(bad_parameter,
                        "bogus credit found while decrementing global reference"
                        "count");

                return (it->second -= count);
            }
            
            // TODO: This needs an exception message explaining why it's invalid.
            throw exception(bad_parameter); 

        } catch (std::bad_alloc) {
            throw exception(out_of_memory);
        } catch (hpx::exception) {
            throw;
        } catch (...) {
            throw exception(internal_server_error);
        }
    }
};

}}}

#endif // HPX_63C75B03_87A6_428E_99A7_F91027E0D463

