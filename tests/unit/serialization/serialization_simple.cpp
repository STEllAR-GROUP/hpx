//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/serialization/serialize.hpp>

#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>

#include <vector>

namespace foo
{
    class bar
    {
        friend class hpx::serialization::access;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {
            std::cout << "serialize bar\n";
        }
    };

    struct baz
    {
    };

    template <typename Archive>
    void serialize(Archive & ar, baz &, unsigned)
    {
        std::cout << "serialize baz\n";
    }

    struct baq
    {
        template <typename Archive>
        void save(Archive & ar, unsigned) const
        {
            std::cout << "saving baq\n";
        }

        template <typename Archive>
        void load(Archive & ar, unsigned)
        {
            std::cout << "loading baq\n";
        }
        HPX_SERIALIZATION_SPLIT_MEMBER()
    };

    struct bap
    {
    };

    template <typename Archive>
    void save(Archive & ar, bap const &, unsigned)
    {
        std::cout << "saving bap\n";
    }

    template <typename Archive>
    void load(Archive & ar, bap &, unsigned)
    {
        std::cout << "loading bap\n";
    }
    HPX_SERIALIZATION_SPLIT_FREE(bap)
}

HPX_IS_BITWISE_SERIALIZABLE(foo::bap);

int main()
{
    std::vector<char> buffer;
    hpx::serialization::output_archive oarchive(buffer);

    foo::bar b1;
    foo::baz b2;
    foo::baq b3;
    foo::bap b4;
    oarchive << b1 << b2 << b3 << b4;
    oarchive & b1 & b2 & b3 & b4;

    hpx::serialization::input_archive iarchive(buffer);
    iarchive >> b1 >> b2 >> b3 >> b4;
    iarchive & b1 & b2 & b3 & b4;
}
