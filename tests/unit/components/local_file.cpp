//  Copyright (c) 2014 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/io/local_file.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include <string>
#include <vector>

using hpx::io::local_file;
using std::string;
using std::vector;

///////////////////////////////////////////////////////////////////////////////
bool test_local_file_component(hpx::id_type const locality, string const& file_name)
{
    {
        // testing writing operations
        local_file lf = local_file::create(locality);

        lf.open_sync(file_name, "w");
        if (!lf.is_open_sync())
        {
            hpx::cerr << "Faile to open OrangeFS file " << file_name <<
                " for write!" << hpx::endl;
            return false;
        }

        char content1[] = "To be or not to be, ";
        char content2[] = "that is the question. ";
        // -1 to get rid of "\0"
        vector<char> buf1(content1, content1 + sizeof(content1) / sizeof(char) - 1);
        vector<char> buf2(content2, content2 + sizeof(content2) / sizeof(char) - 1);

        ssize_t ss = lf.write_sync(buf1);
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf1.size()));
        ss = lf.write_sync(buf2);
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf2.size()));

        int rd = lf.lseek_sync(0, SEEK_SET);
        HPX_TEST_EQ(rd, 0);

        ss = lf.pwrite_sync(buf1, buf1.size() + buf2.size());
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf1.size()));

        ss = lf.pwrite_sync(buf2, 2 * buf1.size() + buf2.size());
        HPX_TEST_EQ(ss, static_cast<ssize_t>(buf2.size()));

        lf.close_sync();
        HPX_TEST(!lf.is_open_sync());
    }

    {
        // testing reading operations
        local_file lf = local_file::create(locality);

        lf.open_sync(file_name, "r");
        if (!lf.is_open_sync())
        {
            hpx::cerr << "Faile to open OrangeFS file " << file_name <<
                " for read!" << hpx::endl;
            return false;
        }

        char content1[] = "To be or not to be, ";
        char content2[] = "that is the question. ";
        vector<char> buf1(content1, content1 + sizeof(content1) / sizeof(char) - 1);
        vector<char> buf2(content2, content2 + sizeof(content2) / sizeof(char) - 1);

        vector<char> buf = lf.read_sync(buf1.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf1.begin(), buf1.end()));
        buf = lf.read_sync(buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf2.begin(), buf2.end()));

        int rd = lf.lseek_sync(0, SEEK_SET);
        HPX_TEST_EQ(rd, 0);

        buf = lf.pread_sync(buf1.size(), buf1.size() + buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf1.begin(), buf1.end()));

        buf = lf.pread_sync(buf2.size(), 2 * buf1.size() + buf2.size());
        HPX_TEST_EQ(string(buf.begin(), buf.end()),
                string(buf2.begin(), buf2.end()));

        lf.close_sync();
        HPX_TEST(!lf.is_open_sync());

        // remove file
        HPX_TEST_EQ(lf.remove_file_sync(file_name), 0);
    }
    return true;
}


///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    int count = 0;
    BOOST_FOREACH(hpx::id_type const& id, localities)
    {
        // DEBUG
        hpx::cerr<<"running tests on locality "<< count << "\n";
        const string file_name = "test_local_file_" +
            boost::lexical_cast<std::string>(count++);
        HPX_TEST(test_local_file_component(id, file_name));
    }

    return hpx::util::report_errors();
}
