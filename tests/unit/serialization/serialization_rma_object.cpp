//  Copyright (c) 2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
//
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
//
#include <hpx/traits/is_rma_eligible.hpp>

#include <hpx/runtime/parcelset/rma/allocator.hpp>
#include <hpx/runtime/parcelset/rma/rma_object.hpp>
//
#include <hpx/runtime/serialization/rma_object.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
//
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>

template <typename T>
struct A
{
    A() {}

    A(T t) : t_(t) {}
    T t_;

    A & operator=(T t) { t_ = t; return *this; }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & t_;
    }
};

struct B {
    std::array<char, 256> stuff;
};

HPX_IS_RMA_ELIGIBLE(B)


namespace hpx { namespace traits
{
    template <>
    template <typename T>
    struct is_rma_elegible< A<T> > : std::true_type
//        : is_bitwise_serializable<char>
    {};
}}


// create and rma vector type for convenience
//template <typename T>
//using rma_vector = hpx::parcelset::rma::rma_object<std::vector<T, rma::allocator<T>>>;
//using namespace hpx::parcelset;

template <typename T>
void test(T min, T max)
{

    {
        std::cout << "Starting " << std::endl;
        // this is where the parcel will be created
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        std::cout << "creating rma vector " << std::endl;
        // fill an rma vector
        rma::rma_vector<T> os;
        std::cout << "reserving " << std::endl;
        os.reserve(max-min);
        std::cout << "filling rma vector " << std::endl;
        for(T c = min; c < max; ++c)
        {
            os.push_back(c);
        }
        std::cout << "serializing rma vector " << std::endl;
        // write rma vector to archive
        oarchive << os;
        std::size_t size = oarchive.bytes_written();
        std::cout << "finished writing rma vector " << std::endl;


        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::cout << "creating in rma vector " << std::endl;
        rma::rma_vector<T> is;
        std::cout << "reading rma vector " << std::endl;

        iarchive >> is;
        HPX_TEST_EQ(os.size(), is.size());
        for(std::size_t i = 0; i < os.size(); ++i)
        {
            HPX_TEST_EQ(os[i], is[i]);
        }
    }
/*
    // try to serialize an rma_object<B>
    {
        // this is where the parcel will be created
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);

        rma::rma_object<A<T>> os = rma::make_rma_object<A<T>>();
        oarchive << os;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        rma::rma_object<A<T>> is;
        iarchive >> is;
//        HPX_TEST_EQ(os.get(), is.get());
    }
*/
/*
    // try to serialize an rma_object<vector<T>>
    {
        // this is where the parcel will be created
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);

        rma::rma_object<B> temp = rma::make_rma_object<B>();
        // fill aan rma vector
//        rma_vector<B> os = hpx::parcelset::rma::make_rma_vector<B>();
//        os->reserve(max-min);
//        for(T c = min; c < max; ++c)
        {
//            os->push_back(c);
        }
        // write rma vector to archive
//        oarchive << os;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::vector<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os->size(), is.size());
        for(std::size_t i = 0; i < os->size(); ++i)
        {
            HPX_TEST_EQ(os->operator[](i), is[i]);
        }
    }
*/

    {
/*
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        // fill aan rma vector
        rma_vector<A<T>> os = hpx::parcelset::rma::make_rma_vector<A<T>>();
        for(T c = min; c < max; ++c)
        {
            os->push_back(c);
        }
//        oarchive << os;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::vector<A<T>> is;
        iarchive >> is;
        HPX_TEST_EQ(os->size(), is.size());
        for(std::size_t i = 0; i < os->size(); ++i)
        {
            HPX_TEST_EQ(os->operator[](i).t_, is[i].t_);
        }
*/
    }
}

template <typename T>
void test_fp(T min, T max)
{
    {
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        std::vector<T> os;
        for(T c = min; c < max; c += static_cast<T>(0.5))
        {
            os->push_back(c);
        }
        oarchive << os;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::vector<T> is;
        iarchive >> is;
        HPX_TEST_EQ(os->size(), is.size());
        for(std::size_t i = 0; i < os->size(); ++i)
        {
            HPX_TEST_EQ(os[i], is[i]);
        }
    }
    {
        std::vector<char> buffer;
        std::vector<hpx::serialization::serialization_chunk> chunks;
        hpx::serialization::output_archive oarchive(buffer, 0, &chunks);
        std::vector<A<T> > os;
        for(T c = min; c < max; c += static_cast<T>(0.5))
        {
            os->push_back(c);
        }
        oarchive << os;
        std::size_t size = oarchive.bytes_written();

        hpx::serialization::input_archive iarchive(buffer, size, &chunks);
        std::vector<A<T> > is;
        iarchive >> is;
        HPX_TEST_EQ(os->size(), is.size());
        for(std::size_t i = 0; i < os->size(); ++i)
        {
            HPX_TEST_EQ(os[i].t_, is[i].t_);
        }
    }
}

int main()
{
    hpx::id_type                    here = hpx::find_here();
    uint64_t                        rank = hpx::naming::get_locality_id_from_id(here);
    std::string                     name = hpx::get_locality_name();
    uint64_t                      nranks = hpx::get_num_localities().get();
    std::size_t                  current = hpx::get_worker_thread_num();
    std::vector<hpx::id_type>    remotes = hpx::find_remote_localities();
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    //
    char const* msg = "hello world from OS-thread %1% on locality "
        "%2% rank %3% hostname %4%";
    std::cout << (boost::format(msg) % current % hpx::get_locality_id()
        % rank % name.c_str()) << std::endl;

    test<char>((std::numeric_limits<char>::min)(),
        (std::numeric_limits<char>::max)());
/*
    test<int>((std::numeric_limits<int>::min)(),
        (std::numeric_limits<int>::min)() + 100);
    test<int>((std::numeric_limits<int>::max)() - 100,
        (std::numeric_limits<int>::max)());
    test<int>(-100, 100);
    test<unsigned>((std::numeric_limits<unsigned>::min)(),
        (std::numeric_limits<unsigned>::min)() + 100);
    test<unsigned>((std::numeric_limits<unsigned>::max)() - 100,
        (std::numeric_limits<unsigned>::max)());
    test<long>((std::numeric_limits<long>::min)(),
        (std::numeric_limits<long>::min)() + 100);
    test<long>((std::numeric_limits<long>::max)() - 100,
        (std::numeric_limits<long>::max)());
    test<long>(-100, 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::min)(),
        (std::numeric_limits<unsigned long>::min)() + 100);
    test<unsigned long>((std::numeric_limits<unsigned long>::max)() - 100,
        (std::numeric_limits<unsigned long>::max)());
    test_fp<float>((std::numeric_limits<float>::min)(),
        (std::numeric_limits<float>::min)() + 100);
    test_fp<float>((std::numeric_limits<float>::max)() - 100,
        (std::numeric_limits<float>::max)()); //it's incorrect
    // because floatmax() - 100 causes cancellations error, digits are not affected
    test_fp<float>(-100, 100);
    test<double>((std::numeric_limits<double>::min)(),
        (std::numeric_limits<double>::min)() + 100);
    test<double>((std::numeric_limits<double>::max)() - 100,
        (std::numeric_limits<double>::max)()); //it's the same
    test<double>(-100, 100);
*/
    return hpx::util::report_errors();
}
