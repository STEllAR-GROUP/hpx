//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SHMEM_DATA_BUFFER_NOV_25_2012_0854PM)
#define HPX_PARCELSET_SHMEM_DATA_BUFFER_NOV_25_2012_0854PM

#include <hpx/hpx_fwd.hpp>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <string>

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
            if (created_)
                boost::interprocess::shared_memory_object::remove(segment_name);
        }
        ~data_buffer_base()
        {
            if (created_)
                boost::interprocess::shared_memory_object::remove(segment_name_.c_str());
        }

        char const* get_segment_name() const { return segment_name_.c_str(); }

    protected:
        std::string segment_name_;
        bool created_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class data_buffer
    {
        typedef boost::interprocess::allocator<
            char, boost::interprocess::managed_shared_memory::segment_manager
        > shmen_allocator_type;

        typedef boost::interprocess::vector<
            char, shmen_allocator_type
        > data_buffer_type;

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
                buffer_ = 0;
                if (created_)
                    segment_.destroy<data_buffer_type>("data");
            }

            data_buffer_type& get_buffer()
            {
                return *buffer_;
            }

            std::size_t size() const
            {
                return segment_.get_size();
            }

        private:
            boost::interprocess::managed_shared_memory segment_;
            shmen_allocator_type allocator_;
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

        void close(boost::system::error_code &ec = boost::system::throws)
        {
            if (data_) {
                data_->close();
                data_.reset();
            }
        }

        data_buffer_type& get_buffer()
        {
            return data_->get_buffer();
        }
        data_buffer_type const& get_buffer() const
        {
            return data_->get_buffer();
        }

        std::size_t size() const
        {
            return data_->size();
        }

        char const* get_segment_name() const
        {
            return data_->get_segment_name();
        }

    private:
        boost::shared_ptr<data> data_;
    };
}}}

#endif
