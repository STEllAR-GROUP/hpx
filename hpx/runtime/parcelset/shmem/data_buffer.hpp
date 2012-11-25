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

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace shmem
{
    ///////////////////////////////////////////////////////////////////////////
    // encapsulate shared data buffer
    struct data_buffer_base
    {
        data_buffer_base(std::string const& segment_name, std::size_t size)
          : segment_name_(segment_name),
            segment_(boost::interprocess::create_only, get_segment_name(), size)
        {
            boost::interprocess::shared_memory_object::remove(segment_name_.c_str());
        }
        data_buffer_base(std::string const& segment_name)
          : segment_name_(segment_name),
            segment_(boost::interprocess::open_only, get_segment_name())
        {
            boost::interprocess::shared_memory_object::remove(segment_name_.c_str());
        }

        ~data_buffer_base()
        {
            boost::interprocess::shared_memory_object::remove(segment_name_.c_str());
        }

        char const* get_segment_name() const { return segment_name_.c_str(); }

    protected:
        std::string segment_name_;
        boost::interprocess::managed_shared_memory segment_;
    };

    ///////////////////////////////////////////////////////////////////////////
    class data_buffer : public data_buffer_base
    {
        typedef boost::interprocess::allocator<
            char, boost::interprocess::managed_shared_memory::segment_manager
        > shmen_allocator_type;

        typedef boost::interprocess::vector<
            char, shmen_allocator_type
        > data_buffer_type;

    public:
        data_buffer(std::string const& segment_name, std::size_t size)
          : data_buffer_base(segment_name, size),
            allocator_(segment_.get_segment_manager()),
            buffer_(segment_.construct<data_buffer_type>("data")(allocator_)),
            created_(true)
        {
        }

        data_buffer(std::string const& segment_name)
          : data_buffer_base(segment_name),
            allocator_(segment_.get_segment_manager()),
            buffer_(segment_.find<data_buffer_type>("data").first),
            created_(false)
        {
        }

        ~data_buffer()
        {
            buffer_ = 0;
            if (created_)
                segment_.destroy<data_buffer_type>("data");
        }

        void close(boost::system::error_code &ec = boost::system::throws)
        {
            buffer_ = 0;
            if (created_) 
                segment_.destroy<data_buffer_type>("data");
        }

    private:
        shmen_allocator_type allocator_;
        data_buffer_type* buffer_;
        bool created_;
    };
}}}

#endif
