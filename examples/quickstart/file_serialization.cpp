//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This example demonstrates how the HPX serialization archives could be used
// to directly store/load to/from a file.

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/traits/serialization_access_data.hpp>

#include <stdexcept>
#include <fstream>

struct file_wrapper
{
    file_wrapper(std::string const& name, std::ios_base::openmode mode)
      : stream_(name.c_str(), mode), size_(0)
    {
        if (!stream_.is_open())
            throw std::runtime_error("Couldn't open file: " + name);

        std::fstream::pos_type fsize = stream_.tellg();
        stream_.seekg(0, std::ios::end);
        size_ = stream_.tellg();
        stream_.seekg(fsize, std::ios::beg);
    }

    std::istream& read(char* s, std::streamsize count) const
    {
        return stream_.read(s, count);
    }

    std::ostream& write(char const* s, std::streamsize count)
    {
        return stream_.write(s, count);
    }

    std::size_t size() const
    {
        return size_;
    }

    void resize(std::size_t count)
    {
        size_ += count;
    }

private:
    mutable std::fstream stream_;
    std::size_t size_;
};

namespace hpx { namespace traits
{
    template <>
    struct serialization_access_data<file_wrapper>
      : default_serialization_access_data<file_wrapper>
    {
        static std::size_t size(file_wrapper const& cont)
        {
            return cont.size();
        }

        static void resize(file_wrapper& cont, std::size_t count)
        {
            return cont.resize(cont.size() + count);
        }

        static void write(file_wrapper& cont, std::size_t count,
            std::size_t current, void const* address)
        {
            cont.write(reinterpret_cast<char const*>(address), count);
        }

        // functions related to input operations
        static void read(file_wrapper const& cont, std::size_t count,
            std::size_t current, void* address)
        {
            cont.read(reinterpret_cast<char*>(address), count);
        }
    };
}}

int main(int argc, char* argv[])
{
    std::size_t size = 0;
    std::vector<double> os;
    {
        file_wrapper buffer("file_serialization_test.archive",
            std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        hpx::serialization::output_archive oarchive(buffer);
        for(double c = -100.0; c < +100.0; c += 1.3)
        {
            os.push_back(c);
        }
        oarchive << os;
        size = oarchive.bytes_written();
    }

    {
        file_wrapper buffer("file_serialization_test.archive",
            std::ios_base::in | std::ios_base::binary);
        hpx::serialization::input_archive iarchive(buffer, size);
        std::vector<double> is;
        iarchive >> is;
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            if (os[i] != is[i])
            {
                std::cerr << "Mismatch for element " << i << ":"
                          << os[i] << " != " << is[i] << "\n";
            }
        }
    }
    return 0;
}


