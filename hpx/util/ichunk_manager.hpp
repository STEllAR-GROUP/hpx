//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ICHUNK_MANAGER_JUL_31_2013_0723PM)
#define HPX_UTIL_ICHUNK_MANAGER_JUL_31_2013_0723PM

#include <hpx/util/binary_filter.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    struct erase_icontainer_type
    {
        virtual ~erase_icontainer_type() {}
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void load_binary(void* address, std::size_t count) = 0;
        virtual void load_binary_chunk(void* address, std::size_t count) = 0;
    };

    template <typename Container>
    struct icontainer_type : erase_icontainer_type
    {
        icontainer_type(Container const& cont, std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(),
            decompressed_size_(inbound_data_size),
            chunks_(0), current_chunk_(std::size_t(-1)), current_chunk_size_(0)
        {}

        icontainer_type(Container const& cont,
                std::vector<serialization_chunk> const* chunks,
                std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(),
            decompressed_size_(inbound_data_size),
            chunks_(0), current_chunk_(std::size_t(-1)), current_chunk_size_(0)
        {
            if (chunks && chunks->size() != 0)
            {
                chunks_ = chunks;
                current_chunk_ = 0;
            }
        }

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
                    *static_cast<unsigned char*>(address) = cont_[current_];
                else
                    std::memcpy(address, &cont_[current_], count);
                current_ += count;

                if (chunks_) {
                    current_chunk_size_ += count;

                    // make sure we switch to the next serialization_chunk if necessary
                    std::size_t current_chunk_size = (*chunks_)[current_chunk_].size_;
                    if (current_chunk_size != 0 && current_chunk_size_ >= current_chunk_size)
                    {
                        // raise an error if we read past the serialization_chunk
                        if (current_chunk_size_ > current_chunk_size)
                        {
                            BOOST_THROW_EXCEPTION(
                                boost::archive::archive_exception(
                                    boost::archive::archive_exception::input_stream_error,
                                    "archive data bstream structure mismatch"));
                            return;
                        }
                        ++current_chunk_;
                        current_chunk_size_ = 0;
                    }
                }
            }
        }

        void load_binary_chunk(void* address, std::size_t count)
        {
            if (filter_.get() || chunks_ == 0 || count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD) {
                // fall back to serialization_chunk-less archive
                this->icontainer_type::load_binary(address, count);
            }
            else {
                BOOST_ASSERT(current_chunk_ != std::size_t(-1));
                BOOST_ASSERT((*chunks_)[current_chunk_].type_ == chunk_type_pointer);

                if ((*chunks_)[current_chunk_].size_ != count) 
                {
                    BOOST_THROW_EXCEPTION(
                        boost::archive::archive_exception(
                            boost::archive::archive_exception::input_stream_error,
                            "archive data bstream data chunk size mismatch"));
                    return;
                }

                // unfortunately we can't implement a zero copy policy on the receiving end
                // as the memory was already allocated by the serialization code
                std::memcpy(address, (*chunks_)[current_chunk_].data_.pos_, count);
                ++current_chunk_;
            }
        }

        Container const& cont_;
        std::size_t current_;
        HPX_STD_UNIQUE_PTR<binary_filter> filter_;
        std::size_t decompressed_size_;

        std::vector<serialization_chunk> const* chunks_;
        std::size_t current_chunk_;
        std::size_t current_chunk_size_;
    };
}}}

#endif

