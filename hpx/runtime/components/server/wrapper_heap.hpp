//  Copyright (c) 1998-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_WRAPPER_HEAP_JUN_12_2008_0904AM)
#define HPX_UTIL_WRAPPER_HEAP_JUN_12_2008_0904AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/bind_naming_wrappers.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/itt_notify.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>

#include <memory>
#include <mutex>
#include <new>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
#if HPX_DEBUG_WRAPPER_HEAP != 0
#define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 1

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
#else
#  define HPX_WRAPPER_HEAP_INITIALIZED_MEMORY 0
#endif

    ///////////////////////////////////////////////////////////////////////////////
    template<typename T, typename Allocator, typename Mutex = hpx::lcos::local::spinlock>
    class wrapper_heap
    {
        HPX_NON_COPYABLE(wrapper_heap);

    public:
        typedef T value_type;
        typedef Allocator allocator_type;

#if HPX_DEBUG_WRAPPER_HEAP != 0
        enum guard_value {
            initial_value = 0xcc,           // memory has been initialized
            freed_value = 0xdd,             // memory has been freed
        };
#endif

        typedef Mutex mutex_type;

        typedef std::unique_lock<mutex_type> scoped_lock;

        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;
//         storage_type data;

        enum {
            heap_step = 0xFFFF,               // default initial number of elements
            heap_size = sizeof(storage_type)  // size of one element in the heap
        };

    public:
        explicit wrapper_heap(
            char const* class_name,
#if defined(HPX_DEBUG)
            std::size_t count,
#else
            std::size_t,
#endif
            std::size_t step = static_cast<std::size_t>(-1)
        )
          : pool_(nullptr), first_free_(nullptr), step_(step), size_(0), free_size_(0),
            base_gid_(naming::invalid_gid),
            class_name_(class_name),
#if defined(HPX_DEBUG)
            alloc_count_(0), free_count_(0), heap_count_(count),
#endif
            heap_alloc_function_("wrapper_heap::alloc", class_name),
            heap_free_function_("wrapper_heap::free", class_name)
        {
            util::itt::heap_internal_access hia;

            HPX_ASSERT(sizeof(storage_type) == heap_size);

        // adjust step to reasonable value
            if (static_cast<std::size_t>(-1) == step_ || step_ < heap_step) //-V104
                step_ = heap_step; //-V101
            else
                step_ = ((step_ + heap_step - 1)/heap_step)*heap_step; //-V104

            if (!init_pool())
                throw std::bad_alloc();
        }

        wrapper_heap()
          : pool_(nullptr), first_free_(nullptr),
            step_(heap_step), size_(0), free_size_(0),
            base_gid_(naming::invalid_gid),
#if defined(HPX_DEBUG)
            alloc_count_(0), free_count_(0), heap_count_(0),
#endif
            heap_alloc_function_("wrapper_heap::alloc", "<unknown>"),
            heap_free_function_("wrapper_heap::free", "<unknown>")
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);

            HPX_ASSERT(sizeof(storage_type) == heap_size);
            if (!init_pool())
                throw std::bad_alloc();
        }

        ~wrapper_heap()
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            tidy();
        }

        std::size_t size() const
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            return size_ - free_size_;
        }
        std::size_t free_size() const
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            return free_size_;
        }
        bool is_empty() const
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            return nullptr == pool_;
        }
        bool has_allocatable_slots() const
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            return first_free_ < pool_+size_;
        }

        bool alloc(T** result, std::size_t count = 1)
        {
            util::itt::heap_allocate heap_allocate(
                heap_alloc_function_, result, count*sizeof(storage_type),
                HPX_WRAPPER_HEAP_INITIALIZED_MEMORY);

            scoped_lock l(mtx_);

            if (!ensure_pool(count))
                return false;

#if defined(HPX_DEBUG)
            alloc_count_ += count;
#endif

            value_type* p = static_cast<value_type*>(first_free_->address()); //-V707
            HPX_ASSERT(p != nullptr);

            first_free_ += count;

            HPX_ASSERT(free_size_ >= count);
            free_size_ -= count;

#if HPX_DEBUG_WRAPPER_HEAP != 0
            // init memory blocks
            debug::fill_bytes(p, initial_value, count*sizeof(storage_type));
#endif

            *result = p;
            return true;
        }

        void free(void *p, std::size_t count = 1)
        {
            util::itt::heap_free heap_free(heap_free_function_, p);

#if HPX_DEBUG_WRAPPER_HEAP != 0
            HPX_ASSERT(did_alloc(p));
#endif
            scoped_lock l(mtx_);

#if HPX_DEBUG_WRAPPER_HEAP != 0
            storage_type* p1 = static_cast<storage_type*>(p);

            HPX_ASSERT(nullptr != pool_ && p1 >= pool_);
            HPX_ASSERT(nullptr != pool_ && p1 + count <= pool_ + size_);
            HPX_ASSERT(first_free_ == nullptr || p1 != first_free_);
            HPX_ASSERT(free_size_ + count <= size_);
            // make sure this has not been freed yet
            HPX_ASSERT(!debug::test_fill_bytes(p1->address(), freed_value,
                count*sizeof(storage_type)));

            // give memory back to pool
            debug::fill_bytes(p1->address(), freed_value, sizeof(storage_type));
#else
            HPX_UNUSED(p);
#endif

#if defined(HPX_DEBUG)
            free_count_ += count;
#endif
            free_size_ += count;

            // release the pool if this one was the last allocated item
            test_release(l);
        }
        bool did_alloc (void *p) const
        {
            // no lock is necessary here as all involved variables are immutable
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);
            return nullptr != pool_ && nullptr != p && pool_ <= p && p < pool_ + size_;
        }

        /// \brief Get the global id of the managed_component instance
        ///        given by the parameter \a p.
        ///
        ///
        /// \note  The pointer given by the parameter \a p must have been
        ///        allocated by this instance of a \a wrapper_heap
        naming::gid_type get_gid(util::unique_id_ranges& ids, void* p,
            components::component_type type)
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);

            HPX_ASSERT(did_alloc(p));

            scoped_lock l(mtx_);
            value_type* addr = static_cast<value_type*>(pool_->address());

            if (!base_gid_) {
                naming::gid_type base_gid;

                {
                    // this is the first call to get_gid() for this heap - allocate
                    // a sufficiently large range of global ids
                    util::unlock_guard<scoped_lock> ul(l);
                    base_gid = ids.get_id(step_, naming::address::address_type(addr));

                    // register the global ids and the base address of this heap
                    // with the AGAS
                    if (!applier::bind_range_local(base_gid, step_,
                            naming::address(hpx::get_locality(), type, addr),
                            sizeof(value_type)))
                    {
                        return naming::invalid_gid;
                    }
                }

                // if some other thread has already set the base GID for this
                // heap, we ignore the result
                if (!base_gid_)
                {
                    // this is the first thread succeeding in binding the new gid range
                    base_gid_ = base_gid;
                }
                else
                {
                    // unbind the range which is not needed anymore
                    util::unlock_guard<scoped_lock> ul(l);
                    applier::unbind_range_local(base_gid, step_);
                }
            }

            return base_gid_ + static_cast<boost::uint64_t>(
                static_cast<value_type*>(p) - addr);
        }

        void set_gid(naming::gid_type const& g)
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);

            scoped_lock l(mtx_);
            base_gid_ = g;
        }

        naming::address get_address()
        {
            util::itt::heap_internal_access hia; HPX_UNUSED(hia);

            value_type* addr = static_cast<value_type*>(pool_->address());
            return naming::address(get_locality(),
                 components::get_component_type<typename value_type::type_holder>(),
                 addr);
        }

    protected:
        bool test_release(scoped_lock& lk)
        {
            if (pool_ == nullptr || free_size_ < size_ || first_free_ < pool_+size_)
                return false;
            HPX_ASSERT(free_size_ == size_);

            // unbind in AGAS service
            if (base_gid_) {
                naming::gid_type base_gid = base_gid_;
                base_gid_ = naming::invalid_gid;

                util::unlock_guard<scoped_lock> ull(lk);
                applier::unbind_range_local(base_gid, step_);
            }

            tidy();
            return true;
        }

        bool ensure_pool(std::size_t count)
        {
            if (nullptr == pool_)
                return false;
            if (first_free_ + count > pool_+size_)
                return false;
            return true;
        }

        bool init_pool()
        {
            HPX_ASSERT(size_ == 0);
            HPX_ASSERT(first_free_ == nullptr);

            std::size_t s = step_ * heap_size; //-V104 //-V707
            pool_ = static_cast<storage_type*>(Allocator::alloc(s));
            if (nullptr == pool_)
                return false;

            first_free_ = pool_;
            size_ = s / heap_size; //-V104
            free_size_ = size_;

            LOSH_(info) //-V128
                << "wrapper_heap ("
                << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                << "): init_pool (" << std::hex << pool_ << ")"
                << " size: " << s << ".";

            return true;
        }

        void tidy()
        {
            if (pool_ != nullptr) {
                LOSH_(debug) //-V128
                    << "wrapper_heap ("
                    << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                    << ")"
#if defined(HPX_DEBUG)
                    << ": releasing heap: alloc count: " << alloc_count_
                    << ", free count: " << free_count_
#endif
                    << ".";
                if (free_size_ != size_
#if defined(HPX_DEBUG)
                     || alloc_count_ != free_count_
#endif
                  )
                {
                    LOSH_(warning) //-V128
                        << "wrapper_heap ("
                        << (!class_name_.empty() ? class_name_.c_str() : "<Unknown>")
                        << "): releasing heap (" << std::hex << pool_ << ")"
                        << " with " << size_-free_size_ << " allocated object(s)!";
                }

                Allocator::free(pool_);
                pool_ = first_free_ = nullptr;
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
#if defined(HPX_DEBUG)
        std::size_t alloc_count_;
        std::size_t free_count_;
        std::size_t heap_count_;
#endif

    private:
        util::itt::heap_function heap_alloc_function_;
        util::itt::heap_function heap_free_function_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace one_size_heap_allocators
    {
        ///////////////////////////////////////////////////////////////////////
        // TODO: this interface should conform to the Boost.Pool allocator
        // interface, to maximize code reuse and consistency - wash.
        //
        // simple allocator which gets the memory from the default malloc,
        // but which does not reallocate the heap (it doesn't grow)
        struct fixed_mallocator
        {
            static void* alloc(std::size_t size)
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
                // return nullptr
                return nullptr;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // heap using malloc and friends
    template <typename T, typename Mutex = hpx::lcos::local::spinlock>
    class fixed_wrapper_heap :
        public wrapper_heap<T, one_size_heap_allocators::fixed_mallocator, Mutex>
    {
    private:
        typedef
            wrapper_heap<T, one_size_heap_allocators::fixed_mallocator, Mutex>
        base_type;

    public:
        explicit fixed_wrapper_heap(char const* class_name = "<Unknown>",
                std::size_t count = 0, std::size_t step = std::size_t(-1))
          : base_type(class_name, count, step)
        {}
    };
}}} // namespace hpx::components::detail

#undef HPX_WRAPPER_HEAP_INITIALIZED_MEMORY

#endif
