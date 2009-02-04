//  Copyright (c) 1998-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_WRAPPER_HEAP_LIST_JUN_14_2008_0409PM)
#define HPX_UTIL_WRAPPER_HEAP_LIST_JUN_14_2008_0409PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/one_size_heap_list.hpp>
#include <hpx/util/generate_unique_ids.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail 
{
    ///////////////////////////////////////////////////////////////////////////
    // list of managed_component heaps
    template<typename Heap, typename Mutex = boost::mutex>
    class wrapper_heap_list : public util::one_size_heap_list<Heap, Mutex>
    {
        typedef util::one_size_heap_list<Heap, Mutex> base_type;

    public:
        wrapper_heap_list(component_type type)
          : base_type(get_component_type_name(type))
        {}

        ///
        naming::id_type get_gid(void* p)
        {
            typename Mutex::scoped_lock guard (this->mtx_);

            typedef typename base_type::const_iterator iterator;
            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it) 
            {
                if ((*it)->did_alloc(p)) 
                    return (*it)->get_gid(id_range_, p);
            }
            return naming::invalid_id;
        }

        ///
        bool get_full_address(void* p, naming::full_address& fa)
        {
            typename Mutex::scoped_lock guard (this->mtx_);

            typedef typename base_type::const_iterator iterator;
            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it) 
            {
                if ((*it)->did_alloc(p)) 
                    return (*it)->get_full_address(id_range_, p, fa);
            }
            return false;
        }

    private:
        // dummy structure implementing the Lockable concept
        struct no_mutex
        {
            struct no_lock
            {
                no_lock(no_mutex&) {}
                void lock() {}
                void unlock() {}
            };

            typedef no_lock scoped_lock;
        };

        util::unique_ids<no_mutex> id_range_;
    };

}}} // namespace hpx::components::detail

#endif
