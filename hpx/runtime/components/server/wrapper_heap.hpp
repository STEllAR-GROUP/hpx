//  Copyright (c) 1998-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_WRAPPER_HEAP_JUN_12_2008_0904AM)
#define HPX_UTIL_WRAPPER_HEAP_JUN_12_2008_0904AM

#include <new>
#include <memory>
#include <string>

#include <boost/noncopyable.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/generate_unique_ids.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
#if HPX_DEBUG_WRAPPER_HEAP != 0
    namespace debug
    {
        ///////////////////////////////////////////////////////////////////////
        // Test memory area for being filled as expected
        inline
        bool test_fill_bytes (void *p, unsigned char c, std::size_t cnt)
        {
            unsigned char* uc = (unsigned char*)p;
            for (std::size_t i = 0; i < cnt; ++i) {
                if (*uc++ != c)
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
#endif

    ///////////////////////////////////////////////////////////////////////////////
    template<typename T, typename Allocator>
    class wrapper_heap : private boost::noncopyable
    {
    public:
        typedef T value_type;
        typedef Allocator allocator_type;

#if HPX_DEBUG_WRAPPER_HEAP != 0
        enum guard_value {
            initial_value = 0xcc,           // memory has been initialized
            freed_value = 0xdd,             // memory has been freed
        };
#endif

        typedef hpx::lcos::local::mutex mutex_type;

        typedef typename mutex_type::scoped_lock scoped_lock;

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;
//         storage_type data;

        enum {
            heap_step = 0xFFFF,               // default initial number of elements
            heap_size = sizeof(storage_type)  // size of one element in the heap
        };

    public:
        explicit wrapper_heap(char const* class_name,
                std::size_t count, std::size_t step = static_cast<std::size_t>(-1))
          : pool_(NULL), first_free_(NULL), step_(step), size_(0), free_size_(0),
            base_gid_(naming::invalid_gid),
            class_name_(class_name), alloc_count_(0), free_count_(0),
            heap_count_(count)
        {
            BOOST_ASSERT(sizeof(storage_type) == heap_size);

        // adjust step to reasonable value
            if (static_cast<std::size_t>(-1) == step_ || step_ < heap_step)
                step_ = heap_step;
            else
                step_ = ((step_ + heap_step - 1)/heap_step)*heap_step;

            if (!init_pool())
                throw std::bad_alloc();
        }

        wrapper_heap()
          : pool_(NULL), first_free_(NULL),
            step_(heap_step), size_(0), free_size_(0),
            base_gid_(naming::invalid_gid),
            alloc_count_(0), free_count_(0), heap_count_(0)
        {
            BOOST_ASSERT(sizeof(storage_type) == heap_size);
            if (!init_pool())
                throw std::bad_alloc();
        }

        ~wrapper_heap()
        {
            tidy();
        }

        std::size_t size() const { return size_ - free_size_; }
        std::size_t free_size() const { return free_size_; }
        bool is_empty() const { return NULL == pool_; }
        bool has_allocatable_slots() const { return first_free_ < pool_+size_; }

        bool alloc(T** result, std::size_t count = 1)
        {
            scoped_lock l(mtx_);

            if (!ensure_pool(count))
                return false;

            alloc_count_ += count;

            value_type* p = static_cast<value_type*>(first_free_->address());
            BOOST_ASSERT(p != NULL);

            first_free_ += count;

            BOOST_ASSERT(free_size_ >= count);
            free_size_ -= count;

#if HPX_DEBUG_WRAPPER_HEAP != 0
            // init memory blocks
            debug::fill_bytes(p, initial_value, count*sizeof(storage_type));
#endif

            *result = p;
            return true;
        }

#if HPX_DEBUG_WRAPPER_HEAP
        void free(void *p, std::size_t count = 1)
#else
        void free(void *, std::size_t count = 1)
#endif
        {
            BOOST_ASSERT(did_alloc(p));

            scoped_lock l(mtx_);

#if HPX_DEBUG_WRAPPER_HEAP != 0
            storage_type* p1 = 0;
            p1 = static_cast<storage_type*>(p);

            BOOST_ASSERT(NULL != pool_ && p1 >= pool_);
            BOOST_ASSERT(NULL != pool_ && p1 + count <= pool_ + size_);
            BOOST_ASSERT(first_free_ == NULL || p1 != first_free_);
            BOOST_ASSERT(free_size_ + count <= size_);
            // make sure this has not been freed yet
            BOOST_ASSERT(!debug::test_fill_bytes(p1->address(), freed_value,
                count*sizeof(storage_type)));

            // give memory back to pool
            debug::fill_bytes(p1->address(), freed_value, sizeof(storage_type));
#endif
            free_count_ += count;
            free_size_ += count;

            // release the pool if this one was the last allocated item
            test_release();
        }
        bool did_alloc (void *p) const
        {
            // no lock is necessary here as all involved variables are immutable
            return NULL != pool_ && NULL != p && pool_ <= p && p < pool_ + size_;
        }

        /// \brief Get the global id of the managed_component instance
        ///        given by the parameter \a p.
        ///
        ///
        /// \note  The pointer given by the parameter \a p must have been
        ///        allocated by this instance of a \a wrapper_heap
        naming::gid_type
        get_gid(util::unique_id_ranges& ids, void* p)
        {
            BOOST_ASSERT(did_alloc(p));

            value_type* addr = static_cast<value_type*>(pool_->address());
            if (!base_gid_) {
                scoped_lock l(mtx_);

                // store a pointer to the AGAS client
                hpx::applier::applier& appl = hpx::applier::get_applier();

                // this is the first call to get_gid() for this heap - allocate
                // a sufficiently large range of global ids
                base_gid_ = ids.get_id(appl.here(), appl.get_agas_client(), step_);

                // register the global ids and the base address of this heap
                // with the AGAS
                if (!applier::bind_range(base_gid_, step_,
                      naming::address(appl.here(),
                          components::get_component_type<typename value_type::type_holder>(),
                          addr),
                      sizeof(value_type)))
                {
                    return naming::invalid_gid;
                }
            }
            return base_gid_ + static_cast<boost::uint64_t>((static_cast<value_type*>(p) - addr));
        }

        void set_gid(naming::gid_type const& g)
        {
            scoped_lock l(mtx_);
            base_gid_ = g;
        }

        naming::address get_address()
        {
            value_type* addr = static_cast<value_type*>(pool_->address());
            return naming::address
                (get_locality(),
                 components::get_component_type<typename value_type::type_holder>(),
                 addr);
        }

    protected:
        bool test_release()
        {
            if (pool_ == NULL || free_size_ < size_ || first_free_ < pool_+size_)
                return false;
            BOOST_ASSERT(free_size_ == size_);

            // unbind in AGAS service
            if (base_gid_) {
                applier::unbind_range(base_gid_, step_);
                base_gid_ = naming::invalid_gid;
            }

            tidy();
            return true;
        }

        bool ensure_pool(std::size_t count)
        {
            if (NULL == pool_)
                return false;
            if (first_free_ + count > pool_+size_)
                return false;
            return true;
        }

        bool init_pool()
        {
            BOOST_ASSERT(size_ == 0);
            BOOST_ASSERT(first_free_ == NULL);

            std::size_t s = step_ * heap_size;
            pool_ = static_cast<storage_type*>(Allocator::alloc(s));
            if (NULL == pool_)
                return false;

            first_free_ = pool_;
            size_ = s / heap_size;
            free_size_ = size_;

            LOSH_(info)
                << "wrapper_heap ("
                << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                << "): init_pool (" << std::hex << pool_ << ")"
                << " size: " << s << ".";

            return true;
        }

        void tidy()
        {
            if (pool_ != NULL) {
                LOSH_(debug)
                    << "wrapper_heap ("
                    << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                    << "): releasing heap: alloc count: " << alloc_count_
                    << ", free count: " << free_count_ << ".";

                if (free_size_ != size_ || alloc_count_ != free_count_) {
                    LOSH_(warning)
                        << "wrapper_heap ("
                        << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                        << "): releasing heap (" << std::hex << pool_ << ")"
                        << " with " << size_-free_size_ << " allocated object(s)!";
                }
                Allocator::free(pool_);
                pool_ = first_free_ = NULL;
                size_ = free_size_ = 0;
            }
        }

    private:
        storage_type* pool_;
        storage_type* first_free_;
        std::size_t step_;
        std::size_t size_;
        std::size_t free_size_;

        // these values are used for AGAS registration of all elements of this
        // managed_component heap
        naming::gid_type base_gid_;

        mutable mutex_type mtx_;

    public:
        std::string const class_name_;
        std::size_t alloc_count_;
        std::size_t free_count_;
        std::size_t heap_count_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace one_size_heap_allocators
    {
        ///////////////////////////////////////////////////////////////////////
        // TODO: this interface should conform to the Boost.Pool allocator
        // inteface/the interface required by <hpx/util/allocator.hpp>, to
        // maximize code reuse and consistency - wash.
        //
        // simple allocator which gets the memory from the default malloc,
        // but which does not reallocate the heap (it doesn't grow)
        struct fixed_mallocator
        {
            static void* alloc(std::size_t& size)
            {
                return ::malloc(size);
            }
            static void free(void* p)
            {
                ::free(p);
            }
            static void* realloc(std::size_t &, void *)
            {
                // normally this should return ::realloc(p, size), but we are
                // not interested in growing the allocated heaps, so we just
                // return NULL
                return NULL;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // heap using malloc and friends
    template<typename T>
    class fixed_wrapper_heap :
        public wrapper_heap<T, one_size_heap_allocators::fixed_mallocator>
    {
    private:
        typedef
            wrapper_heap<T, one_size_heap_allocators::fixed_mallocator>
        base_type;

    public:
        explicit fixed_wrapper_heap(char const* class_name = "<Unknown>",
                std::size_t count = 0, std::size_t step = std::size_t(-1))
          : base_type(class_name, count, step)
        {}
    };

}}} // namespace hpx::components::detail

#endif
