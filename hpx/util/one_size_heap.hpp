//  Copyright (c) 1998-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ONE_SIZE_HEAP_MAY_26_2008_1136AM)
#define HPX_UTIL_ONE_SIZE_HEAP_MAY_26_2008_1136AM

#include <new>
#include <cmemory>
#include <string>

#include <boost/noncopyable.hpp>
#include <hpx/config.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace one_size_heap_allocators
{

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
    namespace debug
    {
        ///////////////////////////////////////////////////////////////////////
        // Test memory area for being filled as expected
        inline 
        bool test_fill_bytes (unsigned char *p, unsigned char c, std::size_t cnt)
        {
            for (std::size_t i = 0; i < cnt; ++i) {
                if (*p++ != c)
                    return false;
            }
            return true;
        }

        ///////////////////////////////////////////////////////////////////////
        // Fill memory area
        inline 
        void fill_bytes (void *p, unsigned char c, int cnt)
        {
            using namespace std;    // some systems have memset in namespace std
            memset (p, c, cnt);
        }

    } // namespace debug
#endif // defined(HPX_DEBUG_ONE_SIZE_HEAP)

    ///////////////////////////////////////////////////////////////////////////////
    template<typename T, typename Allocator>
    class one_size_heap : private boost::noncopyable
    {
    public:
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
        enum guard_value {
            preguard_value = 0xfc,          // values to be used in memory guard areas
            postguard_value = 0xfc,
            initial_value = 0xcc,           // memory has been initialized
            freed_value = 0xdd,             // memory has been free'd
        };
        enum guard_size {
            guard_size = 4,                 // size of guard areas
        };

    private:
        struct data {
            unsigned char preguard_[guard_size];  // guard before data
            union {                         // helper for managing the pool
                unsigned char data_[sizeof(T)];   // object place
                data* next_;                      // pointer to the next free chunk
            } datafield;
            unsigned char postguard_[guard_size]; // guard behind data
            unsigned long alloc_count_;           // add room for allocation counter
        };
#else
        enum guard_size {
            guard_size = 0,                 // size of guard areas
        };
        struct data {
            union {                         // helper for managing the pool
                unsigned char data_[sizeof(T)];   // object place
                data* next_;                      // pointer to the next free chunk
            } datafield;
        };
#endif // defined(HPX_DEBUG_ONE_SIZE_HEAP)

        enum { 
            heap_step = 1024,               // default grow step
            heap_size = sizeof(data)        // size of one element in the heap
        };

    public:
        explicit one_size_heap(char const* class_name, bool throw_on_error = true, 
                bool test_release = true, int step = -1)
          : pool_(NULL), first_free_ (NULL), step_(step), size_(0), free_size_(0), 
            throw_on_error_(throw_on_error), test_release_(test_release)
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
          , class_name_(class_name)
          , alloc_count_(0L), free_count_(0L), heap_count_(0L)
#endif
        {
            BOOST_ASSERT(sizeof(data) == heap_size);

        // adjust step to reasonable value
            if (step_ < heap_step) 
                step_ = heap_step;
            else 
                step_ = ((step_ + heap_step - 1)/heap_step)*heap_step;
        }
        
        one_size_heap()
          : pool_(NULL), first_free_ (NULL), step_(heap_step), size_(0), 
            free_size_(0), throw_on_error_(true), test_release_(true)
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
          , class_name_(""), alloc_count_(0L), free_count_(0L), heap_count_(0L)
#endif
        {
            BOOST_ASSERT(sizeof(data) == heap_size);
        }
        
        ~one_size_heap();
        {
            if (pool_ != NULL) {
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
//             if (free_size_ != size_) {
//                 D_OUTF3(1, 
//                     "one_size_heap: %s:", 
//                     class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>", 
//                     " Releasing heap(%08lx)", pool_, 
//                     " with %ld allocated object(s)!", size_-free_size_);
//             }
#endif
                alloc_.free(pool_);
            }
        }

        int size() const { return size_ - free_size_; }
        int free_size() const { return free_size_; }
        bool is_empty() const { return (NULL == pool_) && test_release_; }

        T* alloc()
        {
            if (!ensure_pool())
                return NULL;

            T *p = static_cast<T*>(&first_free_->datafield.data_);
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            data *item = first_free_;
#endif

            BOOST_ASSERT(p != NULL);
            BOOST_ASSERT(first_free_ != first_free_->datafield.next_);
            BOOST_ASSERT(0 == first_free_->alloc_count_);

            first_free_ = first_free_->datafield.next_;
            free_size_--;

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            // init memory blocks
            debug::fill_bytes(&item->preguard_, preguard_value, sizeof(data.preguard_));
            debug::fill_bytes(&item->datafield.data_, initial_value, sizeof(T));
            debug::fill_bytes(&item->postguard_, postguard_value, sizeof(data.postguard_));
            item->alloc_count_ = ++alloc_count_;
#endif

            return p;
        }
        
        void free (void *p)
        {
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            free_count_++;  // update free counter
            BOOST_ASSERT(alloc_count_ >= free_count_);
#endif

            data* p1 = (data *)((unsigned char *)p - guard_size);

            BOOST_ASSERT(p1->alloc_count_ > 0 && p1->alloc_count_ <= alloc_count_);
            BOOST_ASSERT(debug::test_fill_bytes(&p1->preguard_, preguard_value, guard_size));
            BOOST_ASSERT(debug::test_fill_bytes(&p1->postguard_, postguard_value, guard_size));

            BOOST_ASSERT(NULL != pool_ && p1 >= pool_);
            BOOST_ASSERT(NULL != pool_ && p1 <  pool_ + size_);
            BOOST_ASSERT(first_free_ == NULL || p1 != first_free_);

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            // give memory back to pool
            debug::fill_bytes(p1, freed_value, sizeof(T));
#endif

            p1->datafield.next_ = first_free_;
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
            p1->alloc_count_ = 0;
#endif
            first_free_ = p1;
            free_size_++;
            
            // release the pool if this one was the last allocated item
            test_release();
        }
        bool did_alloc (void *p) const
        {
            return NULL != pool_ && NULL != p && pool_ <= p && p < pool_ + size_;
        }

    protected:
        bool test_release()
        {
            if (pool_ == NULL || free_size_ < size_ || !test_release_)
                return false;
            BOOST_ASSERT(free_size_ == size_);

            alloc_.free(pool_);
            pool_ = first_free_ = NULL;
            size_ = free_size_ = 0;
            return true;
        }

        bool ensure_pool()
        {
            if (pool_ == NULL) {
                if (!init_pool())
                    return false;
            }
            if (first_free_ == NULL) {
                if (!realloc_pool())
                    return false;
            }
            return true;
        }

        bool init_pool()
        {
            BOOST_ASSERT(size_ == 0);
            BOOST_ASSERT(first_free_ == NULL);

            std::size_t s = std::size_t(step_*heap_size);
            pool_ = (data *)alloc_.alloc(s);
            if (NULL == pool_) {
                if (throw_on_error_)
                    throw std::bad_alloc();
                return false;
            }

//         D_OUTF3(2, 
//             "OneSizeHeap: %s:", 
//             class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>",
//             " init_pool (%08lx),", pool_,
//             " size(%ld).", s);

            s /= heap_size;
            first_free_ = pool_;
            size_ = s;
            free_size_ = size_;
            init_pointers();
            return true;
        }

        bool realloc_pool()
        {
            std::size_t delta = std::size_t(step_);
            std::size_t const s = std::size_t((size_ + step_) * heap_size);
            std::size_t s1 = s;

//         D_OUTF3(2, 
//             "OneSizeHeap: %s:", 
//             class_name_.size() > 0 ? class_name_.c_str() : "<Unknown>",
//             " reallocPool(%ld),", s, 
//             " size(%ld).", size_);

            data* p = (data*)alloc_.realloc(s1, pool_);
            if (NULL == p) {
                if (throw_on_error_)
                    throw std::bad_alloc();
                return false;
            }

            BOOST_ASSERT(pool_ == p);
            if (s != s1) {
                delta = (s - s1) / heap_size;
                if (0 == delta) {
                    if (throw_on_error_)
                        throw std::bad_alloc();
                    return false;
                }
            }

            first_free_ = pool_ + size_;
            size_ += delta;
            free_size_ += delta;
            init_pointers();
            return true;
        }

        void init_pointers()
        {
            data *p = first_free_;
            BOOST_ASSERT(NULL != p);
            for (int i = 0; i < step_; ++i, ++p) {
                if (i == step_-1)
                    p->datafield.next_ = NULL;
                else 
                    p->datafield.next_ = p + 1;
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
                p->alloc_count_ = 0;
#endif
            }
        }

    private:
        Allocator alloc_;
        data* pool_;
        data* first_free_;
        int step_;
        int size_;
        int free_size_;
        bool throw_on_error_;
        bool test_release_;

#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
    public:
        std::string class_name_;
        unsigned long alloc_count_;
        unsigned long free_count_;
        unsigned long heap_count_;
#endif
    };

    ///////////////////////////////////////////////////////////////////
    // simple allocator which gets the memory from malloc
    class mallocator  
    {
    public:
        mallocator() {}
        ~mallocator() {}

        static void* alloc(std::size_t& size) 
        { 
            return malloc(size); 
        }
        static void free(void* p) 
        { 
            ::free(p); 
        }
        static void* realloc(std::size_t &size, void *p)
        { 
            return ::realloc(p, size);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // heap using malloc and friends
    template<typename T>
    class one_size_malloc_heap : 
        public one_size_heap<T, one_size_heapAllocators::mallocator>
    {
    private:
        typedef one_size_heap<T, one_size_heapAllocators::mallocator> base_type;
        
    public:
#if defined(HPX_DEBUG_ONE_SIZE_HEAP)
        explicit one_size_malloc_heap(char const* class_name, int step = -1)
          : base_type(class_name, step) 
        {}
#else
        explicit one_size_malloc_heap(int step = -1)
          : base_type(step) 
        {}
#endif
    };

}}} // hpx::util::namespace one_size_heap_allocators

///////////////////////////////////////////////////////////////////////////////
// Macros to minimize typing:
#if defined(HPX_USE_ONESIZEHEAPS)

#define HPX_MAKEUNIQUENAME(x) BOOST_PP_CAT(x, __LINE__)

# if defined(HPX_DEBUG_ONE_SIZE_HEAP) 

///////////////////////////////////////////////////////////////////////////////
// helper macros to declare the new/delete operators inside a class
#define HPX_DECLARE_ONE_SIZE_PRIVATE_HEAP()                                   \
    static void *operator new (std::size_t size,                              \
        char const* filename = "", int line = 0);                             \
    static void operator delete (void* p, std::size_t size);                  \
    /**/

///////////////////////////////////////////////////////////////////////////////
// helper macros for the implementation of one_size_heaps
#define HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP(allocator, dataclass)             \
    namespace {                                                               \
        one_size_heap<dataclass, allocator>                                   \
            HPX_MAKEUNIQUENAME(theHeap)(#dataclass);                          \
    };                                                                        \
    void* dataclass::operator new(std::size_t size, char const* filename,     \
        int nLine)                                                            \
    {                                                                         \
        if (size != sizeof(dataclass))                                        \
            return ::operator new(size, filename, line);                      \
        return HPX_MAKEUNIQUENAME(theHeap).alloc();                           \
    }                                                                         \
    void  dataclass::operator delete (void* p, std::size_t size)              \
    {                                                                         \
        if (NULL == p) return; /* do nothing */                               \
        if (size != sizeof(dataclass)) {                                      \
            ::operator delete(p);                                             \
            return;                                                           \
        }                                                                     \
        HPX_MAKEUNIQUENAME(theHeap).free(p);                                  \
    }                                                                         \
    /**/

# else

///////////////////////////////////////////////////////////////////////////////
// helper macros to declare the new/delete operators inside a class
#define HPX_DECLARE_ONE_SIZE_PRIVATE_HEAP()                                   \
    static void *operator new (std::size_t size);                             \
    static void operator delete (void* p, std::size_t size);                  \
    /**/

///////////////////////////////////////////////////////////////////////////////
// helper macros for the implementation of one_size_heaps
#define HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP(allocator, dataclass)             \
    namespace {                                                               \
        one_size_heap<dataclass, allocator>                                   \
            HPX_MAKEUNIQUENAME(theHeap)(#dataclass);                          \
    };                                                                        \
    void* dataclass::operator new(std::size_t size)                           \
    {                                                                         \
        if (size != sizeof(dataclass))                                        \
            return ::operator new(size);                                      \
        return HPX_MAKEUNIQUENAME(theHeap).alloc();                           \
    }                                                                         \
    void  dataclass::operator delete (void* p, std::size_t size)              \
    {                                                                         \
        if (NULL == p) return; /* do nothing */                               \
        if (size != sizeof(dataclass)) {                                      \
            ::operator delete(p);                                             \
            return;                                                           \
        }                                                                     \
        HPX_MAKEUNIQUENAME(theHeap).free(p);                                  \
    }                                                                         \
    /**/

# endif 

#else 

# define HPX_DECLARE_ONE_SIZE_PRIVATE_HEAP()
# define HPX_IMPLEMENT_ONE_SIZE_PRIVATE_HEAP(allocator, dataclass)

#endif

#endif
