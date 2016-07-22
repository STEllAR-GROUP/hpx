//  Copyright (c) 2013-2014 Hartmut Kaiser
//  Copyright (c) 2015 Andreas Schaefer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_SERIALIZE_BUFFER_APR_05_2013_0312PM)
#define HPX_SERIALIZATION_SERIALIZE_BUFFER_APR_05_2013_0312PM

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/array.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/traits/supports_streaming_with_any.hpp>
#include <hpx/util/bind.hpp>

#include <boost/shared_array.hpp>

#include <algorithm>
#include <type_traits>

namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Allocator = std::allocator<T> >
    class serialize_buffer
    {
    private:
        typedef Allocator allocator_type;

        static void no_deleter(T*) {}

        template <typename Deallocator>
        static void deleter(T* p, Deallocator dealloc, std::size_t size)
        {
            dealloc.deallocate(p, size);
        }

    public:
        enum init_mode
        {
            copy = 0,       // constructor copies data
            reference = 1,  // constructor does not copy data and does not
                            // manage the lifetime of it
            take = 2        // constructor does not copy data but does take
                            // ownership and manages the lifetime of it
        };

        typedef T value_type;

        explicit serialize_buffer(allocator_type const& alloc = allocator_type())
          : size_(0)
          , alloc_(alloc)
        {}

        explicit serialize_buffer(std::size_t size,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            using util::placeholders::_1;
            data_.reset(alloc_.allocate(size),
                        util::bind(&serialize_buffer::deleter<allocator_type>,
                                   _1, alloc_, size_));
        }

        // The default mode is 'copy' which is consistent with the constructor
        // taking a T const * below.
        serialize_buffer (T* data, std::size_t size, init_mode mode = copy,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            if (mode == copy) {
                using util::placeholders::_1;
                data_.reset(alloc_.allocate(size),
                    util::bind(&serialize_buffer::deleter<allocator_type>,
                        _1, alloc_, size_));
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else if (mode == reference) {
                data_ = boost::shared_array<T>(data,
                    &serialize_buffer::no_deleter);
            }
            else {
                // take ownership
                using util::placeholders::_1;
                data_ = boost::shared_array<T>(data,
                    util::bind(&serialize_buffer::deleter<allocator_type>,
                        _1, alloc_, size_));
            }
        }

        template <typename Deallocator>
        serialize_buffer (T* data, std::size_t size,
                allocator_type const& alloc, Deallocator const& dealloc)
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            // if 2 allocators are specified we assume mode 'take'
            using util::placeholders::_1;
            data_ = boost::shared_array<T>(data,
                util::bind(&serialize_buffer::deleter<Deallocator>,
                    _1, dealloc, size_));
        }

        template <typename Deleter>
        serialize_buffer (T* data, std::size_t size, init_mode mode,
                Deleter const& deleter,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            if (mode == copy) {
                data_.reset(alloc_.allocate(size), deleter);
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else {
                // reference or take ownership, behavior is defined by deleter
                data_ = boost::shared_array<T>(data, deleter);
            }
        }

        template <typename Deleter>
        serialize_buffer (T const* data, std::size_t size, init_mode mode, //-V659
                Deleter const& deleter,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            if (mode == copy) {
                data_.reset(alloc_.allocate(size), deleter);
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else if (mode == reference) {
                data_ = boost::shared_array<T>(const_cast<T*>(data), deleter);
            }
            else {
                // can't take ownership of const buffer
                HPX_THROW_EXCEPTION(bad_parameter,
                    "serialize_buffer::serialize_buffer",
                    "can't take ownership of const data");
            }
        }

        // Deleter needs to use deallocator
        template <typename Deallocator, typename Deleter>
        serialize_buffer (T* data, std::size_t size,
                allocator_type const& alloc, Deallocator const& dealloc,
                Deleter const& deleter)
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            // if 2 allocators are specified we assume mode 'take'
            data_ = boost::shared_array<T>(data, deleter);
        }

        // same set of constructors, but taking const data
        serialize_buffer (T const* data, std::size_t size,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            // create from const data implies 'copy' mode
            using util::placeholders::_1;
            data_.reset(alloc_.allocate(size),
                util::bind(&serialize_buffer::deleter<allocator_type>,
                    _1, alloc_, size_));
            if (size != 0)
                std::copy(data, data + size, data_.get());
        }

        template <typename Deleter>
        serialize_buffer (T const* data, std::size_t size,
                Deleter const& deleter,
                allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            // create from const data implies 'copy' mode
            data_.reset(alloc_.allocate(size), deleter);
            if (size != 0)
                std::copy(data, data + size, data_.get());
        }

        serialize_buffer (T const* data, std::size_t size,
                init_mode mode, allocator_type const& alloc = allocator_type())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            if (mode == copy) {
                using util::placeholders::_1;
                data_.reset(alloc_.allocate(size),
                    util::bind(&serialize_buffer::deleter<allocator_type>,
                        _1, alloc_, size_));
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else if (mode == reference) {
                data_ = boost::shared_array<T>(
                    const_cast<T*>(data),
                    &serialize_buffer::no_deleter);
            }
            else {
                // can't take ownership of const buffer
                HPX_THROW_EXCEPTION(bad_parameter,
                    "serialize_buffer::serialize_buffer",
                    "can't take ownership of const data");
            }
        }

        // accessors enabling data access
        T* data() { return data_.get(); }
        T const* data() const { return data_.get(); }

        T* begin() { return data(); }
        T* end() { return data() + size_; }

        T& operator[](std::size_t idx) { return data_[idx]; }
        T operator[](std::size_t idx) const { return data_[idx]; }

        boost::shared_array<T> data_array() const { return data_; }

        std::size_t size() const { return size_; }

    private:
        // serialization support
        friend class hpx::serialization::access;

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const
        {
            ar << size_ << alloc_; //-V128

            if (size_ != 0)
            {
                ar << hpx::serialization::make_array(data_.get(), size_);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void load(Archive& ar, const unsigned int version)
        {
            using util::placeholders::_1;
            ar >> size_ >> alloc_; //-V128

            data_.reset(alloc_.allocate(size_),
                util::bind(&serialize_buffer::deleter<allocator_type>, _1,
                    alloc_, size_));

            if (size_ != 0)
            {
                ar >> hpx::serialization::make_array(data_.get(), size_);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

        // this is needed for util::any
        friend bool
        operator==(serialize_buffer const& rhs, serialize_buffer const& lhs)
        {
            return rhs.data_.get() == lhs.data_.get() && rhs.size_ == lhs.size_;
        }

    private:
        boost::shared_array<T> data_;
        std::size_t size_;
        Allocator alloc_;
    };
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for streaming with util::any, we don't want
    // serialization::serialize_buffer to be streamable
    template <typename T, typename Allocator>
    struct supports_streaming_with_any<serialization::serialize_buffer<T, Allocator> >
      : std::false_type
    {};
}}

#endif
