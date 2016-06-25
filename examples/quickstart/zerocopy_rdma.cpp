//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>

#include <hpx/runtime/serialization/serialize.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
#define ZEROCOPY_DATASIZE   1024*1024

///////////////////////////////////////////////////////////////////////////////
// A custom allocator which takes a pointer in its constructor and then returns
// this pointer in response to any allocate request. It is here to try to fool
// the hpx serialization into copying directly into a user provided buffer
// without copying from a result into another buffer.
template <typename T>
class pointer_allocator
{
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    pointer_allocator() HPX_NOEXCEPT
      : pointer_(nullptr), size_(0)
    {
    }

    pointer_allocator(pointer p, size_type size) HPX_NOEXCEPT
      : pointer_(p), size_(size)
    {
    }

    pointer address(reference value) const { return &value; }
    const_pointer address(const_reference value) const { return &value; }

    pointer allocate(size_type n, void const* hint = nullptr)
    {
        HPX_ASSERT(n == size_);
        return static_cast<T*>(pointer_);
    }

    void deallocate(pointer p, size_type n)
    {
        HPX_ASSERT(p == pointer_ && n == size_);
    }

private:
    // serialization support
    friend class hpx::serialization::access;

    template <typename Archive>
    void load(Archive& ar, unsigned int const version)
    {
        std::size_t t = 0;
        ar >> size_ >> t;
        pointer_ = reinterpret_cast<pointer>(t);
    }

    template <typename Archive>
    void save(Archive& ar, unsigned int const version) const
    {
        std::size_t t = reinterpret_cast<std::size_t>(pointer_);
        ar << size_ << t;
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()

private:
    pointer pointer_;
    size_type size_;
};

///////////////////////////////////////////////////////////////////////////////
// Buffer object used on the client side to specify where to place the received
// data
typedef hpx::serialization::serialize_buffer<double> general_buffer_type;

// Buffer object used for sending the data back to the receiver.
typedef hpx::serialization::serialize_buffer<double, pointer_allocator<double> >
    transfer_buffer_type;

///////////////////////////////////////////////////////////////////////////////
struct zerocopy_server
  : hpx::components::component_base<zerocopy_server>
{
private:
    void release_lock()
    {
        // all we need to do is to unlock the data
        mtx_.unlock();
    }

public:
    zerocopy_server(std::size_t size = 0)
      : data_(size, 3.1415)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    // Retrieve an array of doubles to the given address
    transfer_buffer_type get_here(std::size_t size, std::size_t remote_buffer)
    {
        pointer_allocator<double> allocator(
            reinterpret_cast<double*>(remote_buffer), size);

        // lock the mutex, will be unlocked by the transfer buffer's deleter
        mtx_.lock();

        // we use our data directly without copying
        return transfer_buffer_type(data_.data(), size,
            transfer_buffer_type::reference,
            hpx::util::bind(&zerocopy_server::release_lock, this),
            allocator);
    }
    HPX_DEFINE_COMPONENT_ACTION(zerocopy_server, get_here, get_here_action);

    ///////////////////////////////////////////////////////////////////////////
    // Retrieve an array of doubles
    general_buffer_type get(std::size_t size)
    {
        // lock the mutex, will be unlocked by the transfer buffer's deleter
        mtx_.lock();

        // we use our data directly without copying
        return general_buffer_type(data_.data(), size,
            general_buffer_type::reference,
            hpx::util::bind(&zerocopy_server::release_lock, this));
    }
    HPX_DEFINE_COMPONENT_ACTION(zerocopy_server, get, get_action);

private:
    std::vector<double> data_;
    hpx::lcos::local::spinlock mtx_;
};

typedef hpx::components::component<zerocopy_server> server_type;
HPX_REGISTER_COMPONENT(server_type, zerocopy_server);

typedef zerocopy_server::get_here_action zerocopy_get_here_action;
HPX_REGISTER_ACTION_DECLARATION(zerocopy_get_here_action);
HPX_REGISTER_ACTION(zerocopy_get_here_action);

typedef zerocopy_server::get_action zerocopy_get_action;
HPX_REGISTER_ACTION_DECLARATION(zerocopy_get_action);
HPX_REGISTER_ACTION(zerocopy_get_action);

///////////////////////////////////////////////////////////////////////////////
struct zerocopy
  : hpx::components::client_base<
        zerocopy, hpx::components::stub_base<zerocopy_server> >
{
private:
    // Copy he data once into the destination buffer if the get() operation was
    // entirely local (no data copies have been made so far).
    static void transfer_data(general_buffer_type recv,
        hpx::future<transfer_buffer_type> f)
    {
        transfer_buffer_type buffer(f.get());
        if (buffer.data() != recv.data())
        {
            std::copy(buffer.data(), buffer.data()+buffer.size(), recv.data());
        }
    }

public:
    typedef hpx::components::client_base<
        zerocopy, hpx::components::stub_base<zerocopy_server>
    > base_type;

    zerocopy(hpx::future<hpx::id_type>&& fid)
      : base_type(std::move(fid))
    {}

    //
    hpx::future<void> get_here(general_buffer_type& buff) const
    {
        zerocopy_get_here_action act;

        using hpx::util::placeholders::_1;
        std::size_t buffer_address = reinterpret_cast<std::size_t>(buff.data());
        return hpx::async(act, this->get_id(), buff.size(), buffer_address)
            .then(hpx::util::bind(&zerocopy::transfer_data, buff, _1));
    }
    void get_here_sync(general_buffer_type& buff) const
    {
        get_here(buff).get();
    }

    //
    hpx::future<general_buffer_type> get(std::size_t size) const
    {
        zerocopy_get_action act;
        return hpx::async(act, this->get_id(), size);
    }
    general_buffer_type get_sync(std::size_t size) const
    {
        return get(size).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    for (hpx::id_type const& id : localities)
    {
        zerocopy zc = hpx::new_<zerocopy_server>(id, ZEROCOPY_DATASIZE);

        general_buffer_type buffer(new double[ZEROCOPY_DATASIZE],
            ZEROCOPY_DATASIZE, general_buffer_type::take);

        {
            hpx::util::high_resolution_timer t;

            for (int i = 0; i != 100; ++i)
                zc.get_sync(ZEROCOPY_DATASIZE);

            double d = t.elapsed();
            std::cout << "Elapsed time 'get' (locality "
                        << hpx::naming::get_locality_id_from_id(id)
                        << "): " << d << "[s]\n";
        }

        {
            hpx::util::high_resolution_timer t;

            for (int i = 0; i != 100; ++i)
                zc.get_here_sync(buffer);

            double d = t.elapsed();
            std::cout << "Elapsed time 'get_here' (locality "
                        << hpx::naming::get_locality_id_from_id(id)
                        << "): " << d << "[s]\n";
        }
    }

    return 0;
}

