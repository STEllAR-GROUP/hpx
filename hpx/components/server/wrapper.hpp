//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_WRAPPER_JUN_04_2008_0901PM)
#define HPX_COMPONENTS_WRAPPER_JUN_04_2008_0901PM

#include <boost/throw_exception.hpp>

#include <hpx/exception.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/wrapper_heap.hpp>
#include <hpx/util/one_size_heap_list.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The wrapper template is used as a indirection layer for components 
    /// allowing to gracefully handle the access to non-existing components.
    template <typename Component>
    class wrapper
    {
    protected:
        // takes ownership of the passed pointer
        explicit wrapper(Component* c) 
          : component_(c) 
        {}
        
    public:
        wrapper() 
          : component_(0) 
        {}
        
        ~wrapper()
        {
            delete component_;
        }
        
        ///////////////////////////////////////////////////////////////////////
        // memory management
    protected:
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef 
            util::one_size_heap_list<util::fixed_wrapper_heap<wrapper> > 
        heap_type;

        struct wrapper_heap_tag {};
        
        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            util::static_<heap_type, wrapper_heap_tag> heap;
            return heap.get();
        }

    public:
        /// The memory for wrapper objects is managed by a class specific 
        /// allocator 
        static void* operator new(std::size_t size)
        {
            if (size > sizeof(wrapper))
                return ::operator new(size);
            return get_heap().alloc();
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p) 
                return;     // do nothing if given a NULL pointer
                
            if (size != sizeof(wrapper)) {
                ::operator delete(p);
                return;
            }
            get_heap().free(p);
        }
        
        ///////////////////////////////////////////////////////////////////////
        // The wrapper behaves just like the object it wraps
        Component* operator-> ()
        {
            if (0 == component_)
            {
                boost::throw_exception(hpx::exception(invalid_status));
            }
            return component_;
        }
        
        Component const* operator-> () const
        {
            if (0 == component_)
            {
                boost::throw_exception(hpx::exception(invalid_status));
            }
            return component_;
        }
        
        ///////////////////////////////////////////////////////////////////////
        Component& operator* ()
        {
            if (0 == component_)
            {
                boost::throw_exception(hpx::exception(invalid_status));
            }
            return *component_;
        }
        
        Component const& operator* () const
        {
            if (0 == component_)
            {
                boost::throw_exception(hpx::exception(invalid_status));
            }
            return *component_;
        }
        
    private:
        Component* component_;
    };
    
}}

#endif


