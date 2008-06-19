//  Copyright (c) 1998-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_WRAPPER_HEAP_LIST_JUN_14_2008_0408PM)
#define HPX_UTIL_WRAPPER_HEAP_LIST_JUN_14_2008_0408PM

#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/one_size_heap_list.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail 
{
    ///////////////////////////////////////////////////////////////////////////
    // list of wrapper heaps
    template<typename Heap, typename Mutex = boost::mutex>
    class wrapper_heap_list : public util::one_size_heap_list<Heap, Mutex>
    {
        typedef util::one_size_heap_list<Heap, Mutex> base_type;

    public:
        wrapper_heap_list(char const* class_name = "", int step = -1)
          : base_type(class_name, step)
        {}

        ///
        naming::id_type get_gid(void* p) const
        {
            typedef typename base_type::const_iterator iterator;
            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it) 
            {
                if ((*it)->did_alloc(p)) 
                    return (*it)->get_gid(p);
            }
            return naming::invalid_id;
        }
    };
    
}}} // namespace hpx::components::detail

#endif
