//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ICHUNK_MANAGER_JUL_31_2013_0723PM)
#define HPX_UTIL_ICHUNK_MANAGER_JUL_31_2013_0723PM

#include <boost/version.hpp>
#include <hpx/util/binary_filter.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct erase_icontainer_type
    {
        virtual ~erase_icontainer_type() {}
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void load_binary(void* address, std::size_t count) = 0;
        virtual void load_binary_chunk(void*& address) = 0;
    };

    template <typename Container>
    struct icontainer_type : erase_icontainer_type
    {
        icontainer_type(Container& cont, std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(0),
            decompressed_size_(inbound_data_size),
            chunks_(0), current_chunk_(std::size_t(-1))
        {}

        icontainer_type(Container& cont, std::vector<chunk>* chunks,
                std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(0),
            decompressed_size_(inbound_data_size),
            chunks_(chunks), current_chunk_(chunks ? 0 : std::size_t(-1))
        {}

        ~icontainer_type() {}

        void set_filter(binary_filter* filter)
        {
            filter_.reset(filter);
            if (filter) {
                current_ = filter->init_data(&cont_[current_],
                    cont_.size()-current_, decompressed_size_);

                if (decompressed_size_ < current_)
                {
                    BOOST_THROW_EXCEPTION(
                        boost::archive::archive_exception(
                            boost::archive::archive_exception::input_stream_error,
                            "archive data bstream is too short"));
                    return;
                }
            }
        }

        void load_binary(void* address, std::size_t count)
        {
            if (filter_.get()) {
                filter_->load(address, count);
            }
            else {
                if (current_+count > cont_.size())
                {
                    BOOST_THROW_EXCEPTION(
                        boost::archive::archive_exception(
                            boost::archive::archive_exception::input_stream_error,
                            "archive data bstream is too short"));
                    return;
                }

                if (count == 1)
                    *static_cast<unsigned char*>(address) = buffer_[current_];
                else
                    std::memcpy(address, &cont_[current_], count);
                current_ += count;

                // make sure there switch to the next chunk if necessary
                if (chunks_ && (*chunks_)[current_chunk_].size_ >= current_)
                {
                    if ((*chunks_)[current_chunk_].size_ > current_)
                    {
                        BOOST_THROW_EXCEPTION(
                            boost::archive::archive_exception(
                                boost::archive::archive_exception::input_stream_error,
                                "archive data bstream structure mismatch"));
                        return;
                    }
                    ++current_chunk_;
                }
            }
        }

        void load_binary_chunk(void*& address)
        {
            if (filter_.get() || chunks_ == 0) {
                // fall back to chunk-less archive
                filter_->load(address, count);
            }
            else {
                BOOST_ASSERT(current_chunk_ != std::size_t(-1));

                ++current_chunk_;
                BOOST_ASSERT(current_chunk_ < chunks_->size());
                BOOST_ASSERT((*chunks_)[current_chunk_].type_ == chunk_type_pointer);

                address = (*chunks_)[current_chunk_].data_.pos_;
            }
        }

        Container& cont_;
        std::size_t current_;
        HPX_STD_UNIQUE_PTR<binary_filter> filter_;
        std::size_t decompressed_size_;

        std::vector<chunk>* chunks_;
        std::size_t current_chunk_;
    };
}}}

#endif

