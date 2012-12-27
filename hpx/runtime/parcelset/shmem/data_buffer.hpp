//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Part of the code below has been taken directly from Boost.Container. The 
// original copyright is:
//
// (C) Copyright Ion Gaztanaga 2008-2012. 


#if !defined(HPX_PARCELSET_SHMEM_DATA_BUFFER_NOV_25_2012_0854PM)
#define HPX_PARCELSET_SHMEM_DATA_BUFFER_NOV_25_2012_0854PM

#include <hpx/hpx_fwd.hpp>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#if defined(BOOST_WINDOWS)
#include <boost/interprocess/managed_windows_shared_memory.hpp>
#else
#include <boost/interprocess/managed_shared_memory.hpp>
#endif
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/assert.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
#if defined(BOOST_WINDOWS)
    typedef boost::interprocess::allocator<
        char, boost::interprocess::managed_windows_shared_memory::segment_manager
    > shmem_allocator_type;
#else
    typedef boost::interprocess::allocator<
        char, boost::interprocess::managed_shared_memory::segment_manager
    > shmem_allocator_type;
#endif
}}}

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace container { namespace container_detail 
{
#if BOOST_VERSION < 105300
    // This class template will adapt default construction insertions to 
    // advanced_insert_aux_int
    //
    // We provide the specialization for 'char' to implement proper 
    // uninitialized expansion of the vectors we use below.
    template <>
    struct default_construct_aux_proxy<hpx::parcelset::shmem::shmem_allocator_type, char*>
      : public advanced_insert_aux_int<char*>
    {
        typedef hpx::parcelset::shmem::shmem_allocator_type allocator_type;
        typedef char* iterator_type;

        typedef ::boost::container::allocator_traits<allocator_type> alloc_traits;
        typedef allocator_traits<allocator_type>::size_type size_type;
        typedef allocator_traits<allocator_type>::value_type value_type;
        typedef advanced_insert_aux_int<iterator_type>::difference_type difference_type;

        default_construct_aux_proxy(allocator_type &a, size_type count)
          : a_(a), count_(count)
        {}

        virtual ~default_construct_aux_proxy()
        {}

        virtual void copy_remaining_to(iterator_type)
        {
            // This should never be called with any count
            BOOST_ASSERT(this->count_ == 0);
        }

        virtual void uninitialized_copy_remaining_to(iterator_type p)
        {
            this->priv_uninitialized_copy(p, this->count_);
        }

        virtual void uninitialized_copy_some_and_update(iterator_type pos, 
            difference_type division_count, bool first_n)
        {
            size_type new_count;
            if(first_n) {
                new_count = division_count;
            }
            else {
                BOOST_ASSERT(difference_type(this->count_)>= division_count);
                new_count = this->count_ - division_count;
            }
            this->priv_uninitialized_copy(pos, new_count);
        }

        virtual void copy_some_and_update(iterator_type, 
            difference_type division_count, bool first_n)
        {
            BOOST_ASSERT(this->count_ == 0);
            size_type new_count;
            if(first_n) {
                new_count = division_count;
            }
            else {
                BOOST_ASSERT(difference_type(this->count_) >= division_count);
                new_count = this->count_ - division_count;
            }
            //This function should never called with a count different to zero
            BOOST_ASSERT(new_count == 0);
            (void)new_count;
        }

    private:
        void priv_uninitialized_copy(iterator_type p, const size_type n)
        {
            BOOST_ASSERT(n <= this->count_);

            // We leave memory uninitialized here, which is what should have 
            // happened for 'char' in the first place.
//             iterator_type orig_p = p;
//             size_type i = 0;
//             try {
//                 for(; i < n; ++i, ++p){
//                     alloc_traits::construct(this->a_, container_detail::to_raw_pointer(&*p));
//                 }
//             }
//             catch(...) {
//                 while(i--) {
//                     alloc_traits::destroy(this->a_, container_detail::to_raw_pointer(&*orig_p++));
//                 }
//                 throw;
//             }
            this->count_ -= n;
        }

        allocator_type &a_;
        size_type count_;
    };
#else
    template <>
    struct insert_default_constructed_n_proxy<
        hpx::parcelset::shmem::shmem_allocator_type, char*>
    {
        typedef hpx::parcelset::shmem::shmem_allocator_type allocator_type;
        typedef char* iterator_type;

        typedef ::boost::container::allocator_traits<allocator_type> alloc_traits;
        typedef allocator_traits<allocator_type>::size_type size_type;
        typedef allocator_traits<allocator_type>::value_type value_type;


        explicit insert_default_constructed_n_proxy(allocator_type &a)
          : a_(a)
        {}

        void uninitialized_copy_n_and_update(iterator_type p, size_type n)
        {
            // We leave memory uninitialized here, which is what should have 
            // happened for 'char' in the first place.
//             Iterator orig_p = p;
//             size_type n_left = n;
//             BOOST_TRY{
//                for(; n_left--; ++p){
//                   alloc_traits::construct(this->a_, container_detail::to_raw_pointer(&*p));
//                }
//             }
//             BOOST_CATCH(...){
//                for(; orig_p != p; ++orig_p){
//                   alloc_traits::destroy(this->a_, container_detail::to_raw_pointer(&*orig_p++));
//                }
//                BOOST_RETHROW
//             }
//             BOOST_CATCH_END
        }

        void copy_n_and_update(iterator_type, size_type)
        {
            BOOST_ASSERT(false);
        }

    private:
        allocator_type &a_;
    };
#endif
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    ///////////////////////////////////////////////////////////////////////////
    // encapsulate shared data buffer
    struct data_buffer_base
    {
        data_buffer_base() {}

        data_buffer_base(char const* segment_name, bool created)
          : segment_name_(segment_name),
            created_(created)
        {
#if !defined(BOOST_WINDOWS)
            if (created_)
                boost::interprocess::shared_memory_object::remove(segment_name);
#endif
        }
        ~data_buffer_base()
        {
#if !defined(BOOST_WINDOWS)
            if (created_)
                boost::interprocess::shared_memory_object::remove(segment_name_.c_str());
#endif
        }

        char const* get_segment_name() const { return segment_name_.c_str(); }

    protected:
        std::string segment_name_;
        bool created_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class data_buffer
    {
        typedef boost::interprocess::vector<
            char, shmem_allocator_type> data_buffer_type;

        struct data : public data_buffer_base
        {
            data(char const* segment_name, std::size_t size)
              : data_buffer_base(segment_name, true),
                segment_(boost::interprocess::create_only, segment_name, size + 512),
                allocator_(segment_.get_segment_manager()),
                buffer_(segment_.construct<data_buffer_type>("data")(allocator_))
            {
            }

            data(char const* segment_name)
              : data_buffer_base(segment_name, false),
                segment_(boost::interprocess::open_only, segment_name),
                allocator_(segment_.get_segment_manager()),
                buffer_(segment_.find<data_buffer_type>("data").first)
            {
            }

            ~data()
            {
                close();
            }

            void close()
            {
                if (created_)
                    segment_.destroy<data_buffer_type>("data");
                buffer_ = 0;
            }

            data_buffer_type& get_buffer()
            {
                return *buffer_;
            }

            std::size_t segment_size() const
            {
                return segment_.get_size();
            }

            std::size_t size() const
            {
                return buffer_->size();
            }

            void resize(std::size_t size) const
            {
                return buffer_->resize(size);
            }

        private:
#if defined(BOOST_WINDOWS)
            boost::interprocess::managed_windows_shared_memory segment_;
#else
            boost::interprocess::managed_shared_memory segment_;
#endif
            shmem_allocator_type allocator_;
            data_buffer_type* buffer_;
        };

    public:
        data_buffer()
        {}

        data_buffer(char const* segment_name, std::size_t size)
          : data_(boost::make_shared<data>(segment_name, size))
        {
        }

        data_buffer(char const* segment_name)
          : data_(boost::make_shared<data>(segment_name))
        {
        }

        data_buffer_type& get_buffer()
        {
            return data_->get_buffer();
        }
        data_buffer_type const& get_buffer() const
        {
            return data_->get_buffer();
        }

        std::size_t segment_size() const
        {
            return data_->segment_size();
        }

        std::size_t size() const
        {
            return data_->size();
        }

        void resize(std::size_t size) const
        {
            return data_->resize(size);
        }

        char const* get_segment_name() const
        {
            return data_->get_segment_name();
        }

        void reset()
        {
            data_.reset();
        }

    private:
        boost::shared_ptr<data> data_;
    };
}}}

#endif
