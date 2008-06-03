//  Copyright (c) 1998-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ONE_SIZE_HEAP_LIST_MAY_26_2008_0112PM)
#define HPX_UTIL_ONE_SIZE_HEAP_LIST_MAY_26_2008_0112PM

#include <list>
#include <string>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

#include <hpx/config.hpp>
#include <hpx/util/one_size_heap.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    /////////////////////////////////////////////////////////////////////////////
    //  a list of one_size_heap's
    template<typename T, typename Allocator, typename Mutex = boost::mutex>
    class one_size_heap_list 
    {
    public:
        typedef one_size_heap<T, Allocator> heap_type;
        typedef typename heap_type::data data_type;
        typedef 
            std::list<boost::shared_ptr<one_size_heap<T, Allocator> > > 
        list_type;
        typedef typename list_type::iterator iterator;

        enum { 
            heap_step = heap_type::heap_step,   // default grow step
            heap_size = heap_type::heap_size    // size of the object
        };

    public:
    // ctor/dtor
        explicit one_size_heap_list(char const* class_name, int step = -1)
          : step_(step)
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
          , class_name_(class_name), alloc_count_(0L), free_count_(0L)
          , heap_count_(0L), max_alloc_count_(0L)
#endif
        {
            BOOST_ASSERT(sizeof(typename heap_type::data) == heap_size);

            // adjust step to reasonable value
            if (step_ < heap_step) 
                step_ = heap_step;
            else 
                step_ = ((step_ + heap_step - 1) / heap_step) * heap_step;
        }

        ~one_size_heap_list()
        {
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
//         D_OUTF5(1, 
//             "one_size_heap_list: %s:\r\n", 
//             class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>", 
//             "\tmax count: %ld", max_alloc_count_,
//             " (in %ld heaps),", heap_count_,
//             " alloc count: %ld,", alloc_count_,
//             " free count: %ld.", free_count_);
//                
//         if (alloc_count_ != free_count_) 
//         {
//             D_OUTF1(1, "\treleasing heaplist with %ld allocated object(s)!", alloc_count_-free_count_);
//         }
#endif
        }
        
        // operations
        T* alloc()
        {
            typename Mutex::scoped_lock guard (mtx_);

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            alloc_count_++;
            if (alloc_count_-free_count_ > max_alloc_count_)
                max_alloc_count_ = alloc_count_-free_count_;
#endif

            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it) {
                T *p = NULL;
                
                try {
                    p = (*it)->alloc();
                }
                catch (std::bad_alloc const&) {
                    /**/;
                }
                if (NULL != p) {
                    // will be used as first heap in the future
                    if (it != heap_list_.begin())
                        heap_list_.splice(heap_list_.begin(), heap_list_, it);  
                    return p;
                }
//             else {
//                 D_OUTF4(2, 
//                     "one_size_heap_list: %s:", 
//                     class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>", 
//                     " Failed to allocate from heap(%08lx),", (*it)->heap_count_,
//                     " alloc(%ld),", (*it)->Size(),
//                     " free(%ld).", (*it)->FreeSize());
//             }
            }

            // create new heap
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            iterator itnew = heap_list_.insert(heap_list_.begin(),
                typename list_type::value_type(
                    new heap_type(class_name_.c_str(), false, true, step_)));

            if (itnew == heap_list_.end())
                throw std::bad_alloc();   // insert failed

            (*itnew)->heap_count_ = ++heap_count_;
//         D_OUTF3(2, 
//             "one_size_heap_list: %s:", 
//             class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>", 
//             " Creating new heap(%ld),", heap_count_,
//             " size: %ld.", heap_list_.size());
#else
            iterator itnew = heap_list_.insert(heap_list_.begin(),
                typename list_type::value_type(new heap_type("<Unknown>", 
                    false, true, step_)));

            if (itnew == heap_list_.end())
                throw std::bad_alloc();   // insert failed
#endif

            T* p = (*itnew)->alloc();
            if (NULL == p)
                throw std::bad_alloc();   // snh 
            return p;
        }

        void free(void* p)
        {
            typename Mutex::scoped_lock guard (mtx_);

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            ++free_count_;
#endif
            // find heap which allocated this pointer
            iterator it = heap_list_.begin();
            for (/**/; it != heap_list_.end(); ++it) {
                if ((*it)->did_alloc(p)) {
                    (*it)->free(p);

                    if ((*it)->is_empty()) {
//                     D_OUTF2(2, 
//                         "one_size_heap_list: %s:", 
//                         class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>", 
//                         " Freeing empty heap(%ld).", (*it)->heap_count_);
                        heap_list_.erase (it);
                    }
                    else if (it != heap_list_.begin()) {
                        heap_list_.splice (heap_list_.begin(), heap_list_, it);	
                    }
                    return;
                }
            }
            BOOST_ASSERT(it != heap_list_.end());   // no heap found
        }

        bool did_alloc(void* p) const
        {
            typename Mutex::scoped_lock guard (mtx_);
            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it) {
                if ((*it)->did_alloc(p)) 
                    return true;
            }
            return false;
        }

    private:
        int step_;
        Mutex mtx_;
        list_type heap_list_;

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
    public:
        std::string class_name_;
        unsigned long alloc_count_;
        unsigned long free_count_;
        unsigned long heap_count_;
        unsigned long max_alloc_count_;
#endif
    };

}} // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
// Macros to minimize typing:

#if defined(HPX_USE_ONESIZEHEAPS)

#include <boost/preprocessor/cat.hpp>

///////////////////////////////////////////////////////////////////////////////
// helper macros for the implementation of one_size_heap_lists
#define HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP_LIST(allocator, dataclass)        \
    namespace {                                                               \
        hpx::util::one_size_heap_list<dataclass, allocator>                   \
            BOOST_PP_CAT(theHeap, dataclass)(#dataclass);                     \
    };                                                                        \
    void* dataclass::operator new (size_t size)                               \
    {                                                                         \
        if (size != sizeof(dataclass))                                        \
            return ::operator new(size);                                      \
        return BOOST_PP_CAT(theHeap, dataclass).alloc();                      \
    }                                                                         \
    void  dataclass::operator delete (void* p, size_t size)                   \
    {                                                                         \
        if (NULL == p) return; /* do nothing */                               \
        if (size != sizeof(dataclass)) {                                      \
            ::operator delete(p);                                             \
            return;                                                           \
        }                                                                     \
        BOOST_PP_CAT(theHeap, dataclass).free(p);                             \
    }                                                                         \
    /**/

#else

#define HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP_LIST(allocator, dataclass)

#endif

#endif
