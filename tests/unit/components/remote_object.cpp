//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/function.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/components/remote_object/new.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/fusion/container/vector.hpp>
#include <boost/type_traits/is_convertible.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

///////////////////////////////////////////////////////////////////////////////
struct foo
{
    foo() : i(-1) {}
    foo(int i) : i(i) {}

    int i;
};

std::ostream & operator<<(std::ostream & os, foo const & f)
{
    os << "foo : " << hpx::find_here() << " " << f.i;
    return os;
}

///////////////////////////////////////////////////////////////////////////////
struct output
{
    typedef void result_type;

    void operator()(foo const & f) const
    {
        hpx::cout << f << "\n" << hpx::flush;
    }

    template <typename Archive>
    void serialize(Archive &, unsigned)
    {}
};

///////////////////////////////////////////////////////////////////////////////
int cctor_counter = 0;

struct plus
{
    typedef int result_type;

    int i;
    plus() {}
    plus(int i) : i(i) {}

    plus(plus const & p) : i(p.i) { ++cctor_counter; }

    int operator()(foo const & f) const
    {
        return i + f.i;
    }

    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & i;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map &)
{
    {
        typedef hpx::components::object<foo> object_type;
        typedef hpx::lcos::promise<object_type> object_promise_type;
        typedef std::vector<object_promise_type> object_promises_type;
        typedef std::vector<object_type> objects_type;

        std::vector<hpx::naming::id_type> prefixes = hpx::find_all_localities();
        object_promises_type object_promises;

        int count = 0;
        BOOST_FOREACH(hpx::naming::id_type const & prefix, prefixes)
        {
            using hpx::components::new_;
            object_promises.push_back(new_<foo>(prefix));
            object_promises.push_back(new_<foo>(prefix, count++));
        }

        objects_type objects;
        BOOST_FOREACH(object_promise_type const & promise, object_promises)
        {
            objects.push_back(promise.get());
        }

        BOOST_FOREACH(object_type & o, objects)
        {
            hpx::lcos::wait(o <= output());
            cctor_counter = 0;
            hpx::util::function<int(foo const &)> f = plus(9);
            hpx::cout << (o <= f).get() << "\n" << hpx::flush;
            hpx::cout << cctor_counter << "\n" << hpx::flush;
        }
    }

    hpx::finalize();
    return 0;
}

int main(int argc, char **argv)
{
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    return hpx::init(cmdline, argc, argv);
}


