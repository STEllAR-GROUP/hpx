/**
 Boost Logging library

 Author: John Torjo, www.torjo.com

 Copyright (C) 2007 John Torjo (see www.torjo.com for email)

 Distributed under the Boost Software License, Version 1.0.
    (See accompanying file LICENSE_1_0.txt or copy at
          http://www.boost.org/LICENSE_1_0.txt)

 See http://www.boost.org for updates, documentation, and revision history.
 See http://www.torjo.com/log2/ for more details
*/

/* 
    Tests rolling file
    - tests writing to a clean rolling file
    - tests writing to a rolling file that has been partially written to
    - tests writing to a rolling file that is full (that is, all its files are fully written to)
      thus, we should start writing from the first file
*/

#include <boost/test/minimal.hpp>

#include <boost/logging/format.hpp>
#include <boost/logging/writer/ts_write.hpp>
#include <boost/logging/format/destination/rolling_file.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
namespace fs = boost::filesystem;

const char INPUT_FILE_NAME []= "test.cpp";

using namespace boost::logging;

typedef logger_format_write< > log_type;

BOOST_DEFINE_LOG_FILTER(g_log_filter, filter::no_ts ) 
BOOST_DEFINE_LOG(g_l, log_type)

#define L_ BOOST_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() ) 

// whatever we log to the rolling file, we log here too (easy was to find out all the info that was logged)
std::stringstream g_stringstream;

// keeps all we log to the rolling file, after all its files are full
std::stringstream g_after_full;

// when emulating writing to a log file - each block is the equivalent of a file from the rolling file
std::vector<std::string> g_blocks;

const int MAX_SIZE_PER_FILE = 1024 + 512;
const int FILE_COUNT = 5;

const char NEXT_LINE = '\n';

void init_logs() {
    g_l()->writer().add_destination( 
        destination::rolling_file("out.txt", 
            destination::rolling_file_settings()
                .initial_erase(true)
                .max_size_bytes( MAX_SIZE_PER_FILE)
                .file_count( FILE_COUNT)
                .flush_each_time(true)
                .extra_flags(std::ios_base::binary)
            ));
    g_l()->writer().add_formatter( formatter::idx(), "[%] "  );
    g_l()->writer().add_destination( destination::stream(g_stringstream) );
    g_l()->writer().add_destination( destination::cout() );
    g_l()->turn_cache_off();
}

void write_to_clean_rolling_file() {
    // read this .cpp file - every other line is logged (odd lines)
    std::ifstream in(INPUT_FILE_NAME);
    bool enabled = true;
    std::string line;
    while ( std::getline(in, line) ) {
        g_log_filter()->set_enabled(enabled);
        L_ << "line odd " << line << NEXT_LINE;
        enabled = !enabled;
    }
    g_log_filter()->set_enabled(true);
}

void write_to_existing_rolling_file() { 
    g_l()->writer().del_destination( destination::rolling_file("out.txt") );
    g_l()->writer().add_destination( 
        destination::rolling_file("out.txt", 
            destination::rolling_file_settings()
                .initial_erase(false)
                .max_size_bytes( MAX_SIZE_PER_FILE)
                .file_count( FILE_COUNT)
                .flush_each_time(true)
                .extra_flags(std::ios_base::binary)
            ));

    // read this .cpp file - every other line is logged (even lines now)
    std::ifstream in(INPUT_FILE_NAME);
    bool enabled = false;
    std::string line;
    while ( std::getline(in, line) ) {
        g_log_filter()->set_enabled(enabled);
        L_ << "line even " << line << NEXT_LINE;
        enabled = !enabled;
    }
    g_log_filter()->set_enabled(true);
}

// a bit of white-box testing - we need to know the file names - when dealing with a rolling file
std::string file_name(const std::string & name_prefix, int idx) {
    std::ostringstream out; 
    if ( idx > 0)
        out << name_prefix << "." << (idx+1);
    else
        out << name_prefix;
    return out.str();
}

void test_contents() {
    // now, read each file and see if it matches the blocks
    for ( int idx = 0; idx < FILE_COUNT; ++idx) {
        std::ifstream cur_file( file_name("out.txt", idx).c_str() , std::ios_base::in | std::ios_base::binary );
        std::ostringstream out; 
        out << cur_file.rdbuf();
        std::string cur_file_contents = out.str();
        std::string & cur_block = g_blocks[idx];
        BOOST_CHECK( cur_file_contents == cur_block);
    }
}

void test_contents_after_write_to_existing_rolling_file() { 
    //
    // at this point, we've rolled over - that is, some of the very old contents have been overwritten
    
    // we've also written to a stream, to get the contents as they would have been written, if we hadn't rolled over
    std::string all_contents = g_stringstream.str();
    std::istringstream in(all_contents);

    // now, we emulate writing to a rolled file - but we do it in memory
    g_blocks.resize( FILE_COUNT);
    int cur_block = 0;

    std::string line;
    while ( std::getline(in, line, NEXT_LINE) ) {
        g_blocks[cur_block] += line + NEXT_LINE;

        if ( g_blocks[cur_block].size() > MAX_SIZE_PER_FILE) {
            cur_block = (cur_block + 1) % FILE_COUNT;
            if ( g_blocks[cur_block].size() > MAX_SIZE_PER_FILE)
                // we've rolled to a new file - clear it first
                g_blocks[cur_block].clear();
        }
    }

    // now, read each file and see if it matches the blocks
    for ( int idx = 0; idx < FILE_COUNT; ++idx) {
        std::ifstream cur_file( file_name("out.txt", idx).c_str() , std::ios_base::in | std::ios_base::binary );
        std::ostringstream out; 
        out << cur_file.rdbuf();
        std::string cur_file_contents = out.str();
        std::string & cur_block = g_blocks[idx];
        BOOST_CHECK( cur_file_contents == cur_block);
    }
}

void write_to_too_full_rolling_file() { 
    // make sure all files are too full to be appended to
    for ( int idx = 0; idx < FILE_COUNT; ++idx) {
        std::string cur_file_name = file_name("out.txt", idx);
        std::ofstream cur_file( cur_file_name.c_str() , std::ios_base::out | std::ios_base::app | std::ios_base::binary );
        int file_size = (int)fs::file_size( cur_file_name);
        if ( file_size < MAX_SIZE_PER_FILE) {
            std::string dummy(MAX_SIZE_PER_FILE, ' ');
            g_blocks[idx] += dummy;
            cur_file.write( dummy.c_str(), (std::streamsize)dummy.size() );
        }
    }

    // right now, we know for sure that all files are too big - thus, when logging, we should end up writing to first file first
    g_l()->writer().del_destination( destination::rolling_file("out.txt") );
    g_l()->writer().add_destination( 
        destination::rolling_file("out.txt", 
            destination::rolling_file_settings()
                .initial_erase(false)
                .max_size_bytes( MAX_SIZE_PER_FILE)
                .file_count( FILE_COUNT)
                .flush_each_time(true)
                .extra_flags(std::ios_base::binary)
            ));
    // remember what's written starting now
    g_l()->writer().add_destination( destination::stream(g_after_full) );

    //
    // and right now, do some logging

    // read this .cpp file - every Xth line is written
    const int LINE_PERIOD = 6;
    std::ifstream in(INPUT_FILE_NAME);
    int line_idx = 0;
    std::string line;
    while ( std::getline(in, line) ) {
        if ( (line_idx % LINE_PERIOD) == 0)
            L_ << "new line " << line << NEXT_LINE;
        ++line_idx;
    }

}
void test_contents_after_writing_to_full_rolling_file() { 
    std::string last_contents = g_after_full.str();
    std::istringstream in(last_contents);

    // emulate writing to the rolling file, knowing that we should start writing to the first file from the rolling file
    int cur_block = 0;
    // first time we write, we clear the first file
    g_blocks[0].clear();

    std::string line;
    while ( std::getline(in, line, NEXT_LINE) ) {
        g_blocks[cur_block] += line + NEXT_LINE;

        if ( g_blocks[cur_block].size() > MAX_SIZE_PER_FILE) {
            cur_block = (cur_block + 1) % FILE_COUNT;
            if ( g_blocks[cur_block].size() > MAX_SIZE_PER_FILE)
                // we've rolled to a new file - clear it first
                g_blocks[cur_block].clear();
        }
    }

    test_contents();
}

int test_main(int, char *[]) { 
    fs::path::default_name_check( fs::no_check);

    init_logs();
    write_to_clean_rolling_file();
    write_to_existing_rolling_file();
    test_contents_after_write_to_existing_rolling_file();
    write_to_too_full_rolling_file();
    test_contents_after_writing_to_full_rolling_file();
    return 0;
}
