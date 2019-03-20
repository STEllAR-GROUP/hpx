//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_rmavector_HPP
#define HPX_PARCELSET_rmavector_HPP

#include <hpx/config.hpp>
//
#include <hpx/runtime/parcelset/rma/memory_region.hpp>
#include <hpx/runtime/parcelset/rma/memory_pool.hpp>
//
#include <plugins/parcelport/parcelport_logging.hpp>
//
#include <boost/shared_array.hpp>
//
#include <vector>
#include <functional>
#include <cstddef>
#include <utility>
//
namespace hpx {
namespace parcelset {
namespace rma
{

    namespace detail
    {
        template <typename T>
        struct memory_info
        {
            memory_info(memory_region *r, T *s, memory_pool_base *p)
                : region_(r), data_(s), pool_(p) {};
            //
            memory_region    *region_;
            T                *data_;
            memory_pool_base *pool_;
        };
    }

    // this class looks like a vector, but can be initialized from a pointer and size,
    // it is used by the verbs parcelport to pass an rdma memory chunk with received
    // data into the decode parcel buffer routines.
    // it cannot be resized or changed once created and does not delete wrapped memory
    template<typename T, typename Allocator>
    class rmavector
    {
    public:

        using value_type       = T;
        using reference        = T&;
        using const_reference  = const T&;
        using iterator         = T*;
        using const_iterator   = T const*;
        using difference_type  = typename std::vector<T>::difference_type;
        using size_type        = typename std::vector<T>::size_type;
        //
        using allocator_type   = Allocator;
        using region_type      = memory_region;
        //
        using vector_type      = rmavector<T, allocator_type> ;
        using deleter_callback = std::function<void(void)>;
        //
        using memory_info      = std::shared_ptr<detail::memory_info<T>>;
        // internal vars
        memory_info            m_data_;
        size_type              m_size_;
        deleter_callback       m_cb_;
        allocator_type        *m_alloc_;

        // default construct, allocates nothing
        rmavector() : m_size_(0), m_cb_(0), m_alloc_(nullptr)
        {
            print_debug("default construct", 0);
        }

        // construct with a memory pool allocator
        rmavector(allocator_type* alloc) :
            m_size_(0), m_cb_(0), m_alloc_(alloc)
        {
            print_debug("construct alloc", 0);
        }
/*
        // construct from existing memory chunk, provide allocator, deleter etc
        rmavector(T* p, size_type s, deleter_callback cb,
            allocator_type* alloc, region_type *r) :
                m_size_(s), m_cb_(cb), m_alloc_(alloc)
        {
            print_debug("construct pointer", 0);
        }
*/
        // move constructor,
        rmavector(vector_type && other) :
            m_data_(other.m_data_), m_size_(other.m_size_),
            m_cb_(std::move(other.m_cb_)), m_alloc_(other.m_alloc_)
        {
            print_debug("move constructor", 0);
            //
            other.m_data_   = nullptr;
            other.m_size_   = 0;
            other.m_cb_     = nullptr;
            other.m_alloc_  = nullptr;
        }

        // copy construct : by default, rmavector makes shallow copies
        // from assignment or copy constructor
        rmavector(const rmavector & other)
            : m_data_(other.m_data_)
            , m_size_(other.m_size_)
            , m_cb_(other.m_cb_)
            , m_alloc_(nullptr)
        {
            print_debug("copy constructor : making reference_copy", 0);
        }

        // explicitly request a shallow_copy
        // primarily for use in return value optimization of async calls
        rmavector reference_copy() const
        {
            print_debug("making reference_copy", 0);
            //
            rmavector other;
            other.m_data_   = m_data_;
            other.m_size_   = m_size_;
            other.m_cb_     = m_cb_;
            other.m_alloc_  = nullptr;
            return other;
        }

        // destructor : release shared memory internals and cleanup allocator
        ~rmavector()
        {
            print_debug("destructor", 0);
            // m_cb_();

            // force shared pointer to decremement
            m_data_ = nullptr;
            //
            destroy_allocator();
        }

        // move assignment operator
        vector_type & operator=(vector_type && other)
        {
            m_data_   = other.m_data_;
            m_size_   = other.m_size_;
            m_cb_     = other.m_cb_;
            m_alloc_  = other.m_alloc_;
            //
            print_debug("move assigned", 0);
            //
            other.m_data_   = nullptr;
            other.m_size_   = 0;
            other.m_cb_     = nullptr;
            other.m_alloc_  = nullptr;
            return *this;
        }

        rmavector operator=(vector_type const & other)
        {
            print_debug("ERROR : deep copy assignment", 0);
            HPX_THROW_EXCEPTION(hpx::bad_parameter, "rmavector::operator=",
                "banned for now");
        }

        size_type size() const {
            return m_size_;
        }

        size_type max_size() const {
            return m_size_;
        }

        bool empty() const {
            return m_size_ == 0;
        }

        T *data() {
            return m_data_->data_;
        }

        const T *data() const {
            return m_data_->data_;
        }

        iterator begin() {
            return iterator(&m_data_->data_[0]);
        }

        iterator end() {
            return iterator(&m_data_->data_[m_size_]);
        }

        const_iterator begin() const {
            return iterator(&m_data_->data_[0]);
        }

        const_iterator end() const {
            return iterator(&m_data_->data_[m_size_]);
        }

        reference operator[](size_type index) {
            return m_data_->data_[index];
        }
        const_reference operator[](size_type index) const {
            return m_data_->data_[index];
        }

        void push_back(const T &_Val) {
        }

        size_type capacity() {
            return m_data_->region_ ? m_data_->region_->get_size() : 0;
        }

        void clear() {
            print_debug("clear",0);
            resize(0);
        }

        void reset() {
            print_debug("reset",0);
            // clear size
            resize(0);
            // release anything we might be holding
            m_data_ = nullptr;
            // release allocator
            destroy_allocator();
        }

        void resize(size_type s)
        {
            print_debug("resizing", s);

            // resize data allocation
            reserve(s);

            // if higher size is requested
            if (s > m_size_)
            {
                // default construct the remainder of the new uninitialized memory
                for (size_type i= m_size_; i<s; ++i) {
                    new(&m_data_->data_[i]) T{};
                }
            }
            // if resized smaller
            else {
                // call destructor for unwanted elements
                for (size_type i=s; i<m_size_; ++i) {
                    m_data_->data_[i].~T();
                }
            }
            m_size_ = s;  // change container size.
        }

        void reserve(size_type s)
        {
            create_allocator();
            //
            const size_type bytes = s * sizeof(T);

            // if new size is greater than current region space
            if ((bytes>0) && (!m_data_ || (bytes>m_data_->region_->get_size())))
            {
                print_debug("reserving (bytes)", bytes);

                // allocate a new region
                memory_region *region_ = m_alloc_->allocate_region(bytes);
                T *array_ = static_cast<T*>(region_->get_address());

                // move all previous data. (if m_size_==0, skips loop entirely)
                for (size_type i=0; i<m_size_; ++i) {
                    // call move constructor on new item
                    new(&array_[i]) T(std::move(m_data_->data_[i]));

                    // call the destructor on the moved item
                    m_data_->data_[i].~T();
                }

                // deallocate previous region data if there was any
                m_data_  = nullptr;

                // set shared data to point to our memory/region
                // do not pass a size param as we do not want to change the size
                set_memory_region(region_);

                print_debug("reserved", bytes);
            }
        }

        // only to be used for direct manipulation of internal data
        // such as during serialization
        void set_memory_region(memory_region *region, size_type size=0)
        {
            // release anything we might be holding
            m_data_ = nullptr;
            // make sure an allocator is present
            create_allocator();
            // get the memory pointer from the region
            T *array_ = static_cast<T*>(region->get_address());
            // if requested, set the size of our array
            if (size>0) {
                m_size_ = size;
                // check all is as expected
                HPX_ASSERT(size == region->get_message_length()/sizeof(T));
            }

            // set shared data to point to our memory/region
            m_data_ = std::shared_ptr<detail::memory_info<T>>(
                new detail::memory_info<T>(
                    region, array_, m_alloc_->get_memory_pool()),
                [this](detail::memory_info<T> *mp)
            {
                LOG_TRACE_MSG("rmavector deleter callback");
                if (mp->region_) {
                    LOG_TRACE_MSG("rmavector shared array destructor "
                          << "this "   << hexpointer(this)
                          << "region "   << hexpointer(mp->region_)
                          << "array "    << hexpointer(mp->data_));
                    HPX_ASSERT(mp->data_ == mp->region_->get_address());
                    mp->pool_->release_region(mp->region_);
                    delete mp;
                }
                else {
                    LOG_ERROR_MSG("shared array 2 destructor "
                        << "this " << hexpointer(this));
                }
            });
            //
            print_debug("memory region assigned", size);
        }

        allocator_type get_allocator() const {
            return allocator_type(*m_alloc_);
        }

        region_type *get_region() const {
            return m_data_->region_;
        }

        void create_allocator()
        {
            if (m_alloc_) {
                return;
            }
            //
            parcelset::parcelhandler &ph =
                hpx::get_runtime().get_parcel_handler();
            auto pp = ph.get_default_parcelport();

            rma::allocator<char> *allocator = pp->get_allocator();
            typedef typename
                std::allocator_traits<rma::allocator<char>>::template rebind_alloc<T>
                T_allocator;
            //
            m_alloc_ = new T_allocator(allocator->get_memory_pool());
            print_debug("created allocator", 0);
        }

        inline void destroy_allocator()
        {
            if (!m_alloc_) {
                return;
            }
            delete m_alloc_;
            m_alloc_ = nullptr;
        }

        void deallocate_data() {
            print_debug("deallocate_data", 0);
        }

        template <typename P>
        void print_debug(const char *msg, P p) const {
            LOG_TRACE_MSG("rmavector "
                << "this "     << hexpointer(this)
                << msg << " "  << p << " "
                << "region "   << hexpointer(m_data_ ? m_data_->region_ : nullptr)
                << "array "    << hexpointer(m_data_ ? m_data_->data_ : nullptr)
                << "refcount " << hexnumber(m_data_ ? m_data_.use_count() : 0)
                << "size "     << hexuint32(m_size_)
                << "alloc "    << hexpointer(m_alloc_));
        }
    };
}}}

#endif
