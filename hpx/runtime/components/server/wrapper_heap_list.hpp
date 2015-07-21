//  Copyright (c) 1998-2013 Hartmut Kaiser
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
    template <typename Heap, typename Mutex = lcos::local::spinlock>
    class wrapper_heap_list
      : public util::one_size_heap_list<Heap, Mutex>
    {
        typedef util::one_size_heap_list<Heap, Mutex> base_type;

    public:
        wrapper_heap_list(component_type type)
          : base_type(get_component_type_name(type)),
            type_(type)
        {}

        ///
        naming::gid_type get_gid(void* p)
        {
            typename base_type::unique_lock_type guard(this->mtx_);

            typedef typename base_type::const_iterator iterator;
            iterator end = this->heap_list_.end();
            for (iterator it = this->heap_list_.begin(); it != end; ++it)
            {
                if ((*it)->did_alloc(p))
                {
                    util::unlock_guard<typename base_type::unique_lock_type> ul(guard);
                    return (*it)->get_gid(id_range_, p, type_);
                }
            }
            return naming::invalid_gid;
        }

        void set_range(
            naming::gid_type const& lower
          , naming::gid_type const& upper)
        {
            typename base_type::unique_lock_type guard(this->mtx_);
            id_range_.set_range(lower, upper);
        }

    private:
        util::unique_id_ranges id_range_;
        components::component_type type_;
    };

}}} // namespace hpx::components::detail

#endif
