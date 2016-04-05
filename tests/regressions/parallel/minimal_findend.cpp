//  Copyright (c) 2014 Grant Mercer
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct decorated_iterator
      : boost::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
    {
    private:
        typedef boost::iterator_adaptor<
            decorated_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
        base_type;

    public:
        decorated_iterator()
        {}

        decorated_iterator(BaseIterator base)
          : base_type(base)
        {}

        decorated_iterator(BaseIterator base, std::function<void()> f)
          : base_type(base), m_callback(f)
        {}

    private:
        friend class boost::iterator_core_access;

        typename base_type::reference dereference() const
        {
            if (m_callback)
                m_callback();
            return *(this->base());
        }

    private:
        std::function<void()> m_callback;
    };
}

void find_end_failing_test()
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, std::random_access_iterator_tag>
        decorated_iterator;

    std::vector<std::size_t> c(10007, 0);
    std::size_t h[] = {1,2};

    bool caught_exception = false;
    try {
        std::find_end(
            decorated_iterator(
                boost::begin(c),
                [](){ throw std::runtime_error("error"); }),
            decorated_iterator(
                boost::end(c),
                [](){ throw std::runtime_error("error"); }),
            boost::begin(h), boost::end(h));

        // should never reach this point
        HPX_TEST(false);
    }
    catch(std::runtime_error const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

int hpx_main()
{
    find_end_failing_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exted with non-zero status");

    return hpx::util::report_errors();
}
