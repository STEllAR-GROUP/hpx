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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    /////////////////////////////////////////////////////////////////////////////
    //  a list of one_size_heap's
    template<typename Heap, typename Mutex = boost::mutex>
    class one_size_heap_list 
    {
    public:
        typedef Heap heap_type;

        typedef typename heap_type::allocator_type allocator_type;
        typedef typename heap_type::value_type value_type;

        typedef std::list<boost::shared_ptr<heap_type> > list_type;
        typedef typename list_type::iterator iterator;
        typedef typename list_type::const_iterator const_iterator;

        enum { 
            heap_step = heap_type::heap_step,   // default grow step
            heap_size = heap_type::heap_size    // size of the object
        };

    public:
    // ctor/dtor
        explicit one_size_heap_list(char const* class_name = "")
          : class_name_(class_name), alloc_count_(0L), free_count_(0L),
            heap_count_(0L), max_alloc_count_(0L)
        {
            BOOST_ASSERT(sizeof(typename heap_type::storage_type) == heap_size);
        }

        explicit one_size_heap_list(std::string const& class_name)
          : class_name_(class_name), alloc_count_(0L), free_count_(0L),
            heap_count_(0L), max_alloc_count_(0L)
        {
            BOOST_ASSERT(sizeof(typename heap_type::storage_type) == heap_size);
        }

        ~one_size_heap_list()
        {
            LOSH_(info) 
                << "one_size_heap_list (" 
                << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                << "): releasing heap_list: max count: " << max_alloc_count_ 
                << " (in " << heap_count_ << " heaps), alloc count: " 
                << alloc_count_ << ", free count: " << free_count_ << ".";

            if (alloc_count_ != free_count_) 
            {
                LOSH_(warning) 
                    << "one_size_heap_list (" 
                    << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                    << "): releasing heap_list with " << alloc_count_-free_count_ 
                    << " allocated object(s)!";
            }
        }
        
        // operations
        value_type* alloc(std::size_t count = 1)
        {
            if (0 == count)
                throw std::bad_alloc();   // this doesn't make sense for us

            typename Mutex::scoped_lock guard (mtx_);

            alloc_count_ += (int)count;
            if (alloc_count_-free_count_ > max_alloc_count_)
                max_alloc_count_ = alloc_count_-free_count_;

            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it) {
                value_type *p = NULL;
                
                try {
                    p = (*it)->alloc(count);
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
                else {
                    LOSH_(info) 
                        << "one_size_heap_list (" 
                        << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                        << "): failed to allocate from heap (" << (*it)->heap_count_ 
                        << "), allocated: " << (*it)->size() << ", free'd: " 
                        << (*it)->free_size() << ".";
                }
            }

            // create new heap
            iterator itnew = heap_list_.insert(heap_list_.begin(),
                typename list_type::value_type(
                    new heap_type(class_name_.c_str(), false, true, count)));

            if (itnew == heap_list_.end())
                throw std::bad_alloc();   // insert failed

            (*itnew)->heap_count_ = ++heap_count_;
            LOSH_(info) 
                << "one_size_heap_list (" 
                << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                << "): creating new heap (" << heap_count_ 
                << "), size of heap_list: " << heap_list_.size() << ".";

            value_type* p = (*itnew)->alloc(count);
            if (NULL == p)
                throw std::bad_alloc();   // snh 
            return p;
        }

        void free(void* p, std::size_t count = 1)
        {
            typename Mutex::scoped_lock guard (mtx_);

            free_count_ += (int)count;

            // find heap which allocated this pointer
            iterator it = heap_list_.begin();
            for (/**/; it != heap_list_.end(); ++it) {
                if ((*it)->did_alloc(p)) {
                    (*it)->free(p, count);

                    if ((*it)->is_empty()) {
                        LOSH_(info) 
                            << "one_size_heap_list (" 
                            << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                            << "): freeing empty heap (" << (*it)->heap_count_ << ").";

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

    protected:
        Mutex mtx_;
        list_type heap_list_;

    public:
        std::string class_name_;
        unsigned long alloc_count_;
        unsigned long free_count_;
        unsigned long heap_count_;
        unsigned long max_alloc_count_;
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
        hpx::util::one_size_heap_list<                                        \
            hpx::util::one_size_heap<dataclass, allocator>                    \
        > BOOST_PP_CAT(theHeap, dataclass)(#dataclass);                       \
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
