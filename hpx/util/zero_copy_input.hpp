// Copyright (c) 2012 Bryce Adelstein-Lelbach
// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ZERO_COPY_INPUT_OCT_15_2012_0134PM)
#define HPX_UTIL_ZERO_COPY_INPUT_OCT_15_2012_0134PM

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/foreach.hpp>
#include <boost/integer/endian.hpp>
#include <boost/serialization/vector.hpp>

#include <vector>

namespace hpx { namespace util
{
    struct zero_copy_input
    {
        typedef util::container_device<std::vector<char> > io_device_type;

        zero_copy_input() 
          : in_priority_(0),
            in_chunks_(0),
            current_chunk_(0),
            current_buffer_(0)
        {}

        void prepare()
        {
            clear();

            buffers_.push_back(boost::asio::buffer(&in_priority_, sizeof(in_priority_)));
            buffers_.push_back(boost::asio::buffer(&in_chunks_, sizeof(in_chunks_)));
        }

        void prepare_chunks()
        {
            buffers_.clear();

            chunk_sizes_.resize(in_chunks_);
            buffers_.push_back(boost::asio::buffer(chunk_sizes_));
        }

        void load_pass1()
        {
            buffers_.clear();

            in_buffers_.push_back(std::vector<char>(chunk_sizes_[current_chunk_++]));
            buffers_.push_back(boost::asio::buffer(in_buffers_.back()));
        }

        template <typename T>
        void load_pass2(std::vector<T>& pv)
        {
            typedef container_device<std::vector<char> > io_device_type;
            boost::iostreams::stream<io_device_type> io(
                in_buffers_[current_buffer_++]);

            {
                // Deserialize the data the slow way.
                portable_binary_iarchive archive(io);
                archive >> pv;
            }
        }

        void clear()
        {
            in_priority_ = 0;
            in_chunks_ = 0;

            current_chunk_ = 0;
            chunk_sizes_.clear();

            current_buffer_ = 0;
            buffers_.clear();

            in_buffers_.clear();
        }

        // access the vector of buffers used for serialization
        std::vector<boost::asio::mutable_buffer>& get_buffers() 
        {
            return buffers_;
        }

        boost::integer::ulittle8_t::value_type priority() const
        {
            return in_priority_;
        }

    private:
        boost::integer::ulittle8_t in_priority_;
        boost::integer::ulittle64_t in_chunks_;
        std::size_t current_chunk_;
        std::vector<boost::integer::ulittle64_t> chunk_sizes_;
        std::size_t current_buffer_;
        std::vector<std::vector<char> > in_buffers_;
        std::vector<boost::asio::mutable_buffer> buffers_;
    };
}}

#endif
