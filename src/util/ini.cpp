//  Copyright (c) 2005-2007 Andre Merzky 
//  Copyright (c) 2005-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// System Header Files
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>

#include <list>
#include <vector>
#include <iostream>
#include <fstream>

#include <boost/assert.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/bind.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/ini.hpp>

#ifdef _APPLE
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#elif !defined(BOOST_WINDOWS)
extern char **environ;
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{

///////////////////////////////////////////////////////////////////////////////
// example ini line: line # comment
const char pattern_comment[] =  "^([^#]*)(#.*)$";

// example uses ini line: [sec.ssec]
const char pattern_section[] = "^\\[([^\\]]+)\\]$";

// example uses ini line: key = val
const char pattern_entry[] = "^([^\\s=]+)\\s*=\\s*(.*[^\\s])\\s*$";
// FIXME: allow empty values, interpret as ""

///////////////////////////////////////////////////////////////////////////////

namespace 
{
    ///////////////////////////////////////////////////////////////////////////
    inline std::string
    trim_whitespace (std::string const &s)
    {
        typedef std::string::size_type size_type;

        size_type first = s.find_first_not_of (" \t\r\n");
        if (std::string::npos == first)
            return (std::string ());

        size_type last = s.find_last_not_of (" \t\r\n");
        return s.substr (first, last - first + 1);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline section*
    add_section_if_new(section* current, std::string const& sec_name)
    {
        // do we know this one?
        if (!current->has_section(sec_name)) {
            // no - add it!
            section sec;
            current->add_section (sec_name, sec, current->get_root());
        }
        return current->get_section (sec_name);
    }

} // namespace

///////////////////////////////////////////////////////////////////////////////
section::section (std::string const& filename)
  : root_(this_()), name_(filename)
{
    if (!filename.empty())
        read(filename);
    else
        regex_init();
}

section::section (const section & in)
  : root_(this_()), name_(in.get_name())
{
    regex_init();

    entry_map const& e = in.get_entries();
    entry_map::const_iterator end = e.end();
    for (entry_map::const_iterator i = e.begin (); i != end; ++i)
        add_entry(i->first, i->second);

    section_map s = in.get_sections();
    section_map::iterator send = s.end();
    for (section_map::iterator si = s.begin(); si != send; ++si)
        add_section (si->first, si->second, get_root());
}

void section::read (std::string const& filename)
{
#if defined(__AIX__) && defined(__GNUC__)
    // NEVER ask why... seems to be some weird stdlib initialization problem
    // If you don't call getline() here the while(getline...) loop below will
    // crash with a bad_cast excetion. Stupid AIX...
    std::string l1;
    std::ifstream i1;
    i1.open(filename.c_str(), std::ios::in);
    std::getline(i1, l1);
    i1.close();
#endif

    // build ini - open file and parse each line
    int linenum = 0;
    std::ifstream input(filename.c_str (), std::ios::in);
    if (!input.is_open())
        line_msg("Cannot open file: ", filename);

    // initialize
    if (!regex_init())
        line_msg ("Cannot init regex for ", filename);

    // parse file
    section * current = this;
    std::string line;

    boost::regex regex_comment (pattern_comment, boost::regex::perl | boost::regex::icase);
    boost::regex regex_section (pattern_section, boost::regex::perl | boost::regex::icase);
    boost::regex regex_entry (pattern_entry,   boost::regex::perl | boost::regex::icase);
    while (std::getline (input, line))
    {
        ++linenum;

        // remove trailing new lines and white spaces
        line = trim_whitespace (line);

        // skip if empty line
        if (line.empty())
            continue;

        // weep out comments
        boost::smatch what_comment;
        if (boost::regex_match (line, what_comment, regex_comment))
        {
            BOOST_ASSERT(3 == what_comment.size());

            line = trim_whitespace (what_comment[1]);
            if (line.empty())
                continue;
        }

        // no comments anymore: line is either section, key=val,
        // or garbage/empty
        boost::smatch what;
        if (boost::regex_match(line, what, regex_section))
        {
            // found a section line
            if(2 != what.size())
            {
                line_msg("Cannot parse sec in ", filename, linenum);
            }

            current = this;     // start adding sections at the root

            // got the section name. It might be hierarchical, so split it up, and
            // for each elem, check if we have it.  If not, create it, and add
            std::string sec_name (what[1]);
            std::string::size_type pos = 0;
            for (std::string::size_type pos1 = sec_name.find_first_of('.');
                 std::string::npos != pos1;
                 pos1 = sec_name.find_first_of ('.', pos = pos1 + 1))
            {
                current = add_section_if_new(current, sec_name.substr(pos, pos1 - pos));
            }

            current = add_section_if_new (current, sec_name.substr(pos));
        }

        // did not match section, so might be key/val entry
        else if ( boost::regex_match (line, what, regex_entry) )
        {
            // found a entry line
            if (3 != what.size())
            {
                line_msg("Cannot parse key/value in ", filename, linenum);
            }

            // add key/val to current section
            current->add_entry (what[1], what[2]);
        }
        else {
            // Hmm, is not a section, is not an entry, is not empty - must be an error!
            line_msg ("Cannot parse line at ", filename, linenum);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
void section::add_section (std::string const& sec_name, section& sec, section* root)
{
    // setting name and root
    sec.name_ = sec_name;
    sec.set_root((NULL != root) ? root : get_root());
    sections_[sec_name] = sec;
}

bool section::has_section (std::string const& sec_name) const
{
    std::string::size_type i = sec_name.find(".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0,  i);
        std::string sub_sec_name = sec_name.substr (1 + i);

        section_map::const_iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            return (*it).second.has_section (sub_sec_name);
        }
        return false;
    }
    return sections_.find(sec_name) != sections_.end();
}

section* section::get_section (std::string const& sec_name) 
{
    std::string::size_type i  = sec_name.find (".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0,  i);
        section_map::iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            std::string sub_sec_name = sec_name.substr(i+1);
            return (*it).second.get_section(sub_sec_name);
        }

        std::string name(get_name());
        if (name.empty())
            name = "<root>";

        boost::throw_exception(hpx::exception(bad_parameter, 
            "No such section (" + sec_name + ") in section: " + name));
        return NULL;
    }

    section_map::iterator it = sections_.find(sec_name);
    if (it != sections_.end())
        return &((*it).second);

    boost::throw_exception(hpx::exception(bad_parameter, 
        "No such section (" + sec_name + ") in section: " + get_name()));
    return NULL;
}

section const* section::get_section (std::string const& sec_name) const
{
    std::string::size_type i  = sec_name.find (".");
    if (i != std::string::npos)
    {
        std::string cor_sec_name = sec_name.substr (0,  i);
        section_map::const_iterator it = sections_.find(cor_sec_name);
        if (it != sections_.end())
        {
            std::string sub_sec_name = sec_name.substr(i+1);
            return (*it).second.get_section(sub_sec_name);
        }

        std::string name(get_name());
        if (name.empty())
            name = "<root>";

        boost::throw_exception(hpx::exception(bad_parameter, 
            "No such section (" + sec_name + ") in section: " + name));
        return NULL;
    }

    section_map::const_iterator it = sections_.find(sec_name);
    if (it != sections_.end())
        return &((*it).second);

    boost::throw_exception(hpx::exception(bad_parameter, 
        "No such section (" + sec_name + ") in section: " + get_name()));
    return NULL;
}

void section::add_entry (std::string const& key, std::string val)
{
    if (!val.empty())
        val = expand_entry(val);

    if (val.empty())
        entries_.erase(key);
    else
        entries_[key] = val;
}

bool section::has_entry (std::string const& key) const
{
    std::string::size_type i = key.find (".");
    if (i != std::string::npos)
    {
        std::string sub_sec = key.substr(0, i);
        std::string sub_key = key.substr(i+1, key.size() - i);
        if (has_section(sub_sec))
        {
            section_map::const_iterator cit = sections_.find(sub_sec);
            BOOST_ASSERT(cit != sections_.end());
            return (*cit).second.has_entry(sub_key);
        }
        return false;
    }
    return entries_.find(key) != entries_.end();
}

std::string section::get_entry (std::string const& key) const
{
    std::string::size_type i = key.find (".");
    if (i != std::string::npos)
    {
        std::string sub_sec = key.substr(0, i);
        std::string sub_key = key.substr(i+1, key.size() - i);
        if (has_section(sub_sec))
        {
            section_map::const_iterator cit = sections_.find(sub_sec);
            BOOST_ASSERT(cit != sections_.end());
            return (*cit).second.get_entry(sub_key);
        }

        boost::throw_exception(hpx::exception(bad_parameter, 
            "No such key (" + key + ") in section: " + get_name()));
        return "";
    }

    if (entries_.find(key) != entries_.end())
    {
        entry_map::const_iterator cit = entries_.find(key);
        BOOST_ASSERT(cit != entries_.end());
        return (*cit).second;
    }

    boost::throw_exception(hpx::exception(bad_parameter, 
        "No such section (" + key + ") in section: " + get_name()));
    return "";
}

std::string 
section::get_entry (std::string key, std::string const& default_val) const
{
    typedef std::vector<std::string> string_vector;

    string_vector split_key;
    boost::split(split_key, key, boost::is_any_of("."));

    key = split_key.back();
    split_key.pop_back();

    section const* cur_section = this;
    for (string_vector::const_iterator iter = split_key.begin(),
         end = split_key.end(); iter != end; ++iter)
    {
        section_map::const_iterator next = cur_section->sections_.find(*iter);
        if (cur_section->sections_.end() == next)
            return default_val;
        cur_section = &next->second;
    }

    entry_map::const_iterator entry = cur_section->entries_.find(key);
    if (cur_section->entries_.end() == entry)
        return default_val;

    return entry->second;
}

section section::clone(section* root) const
{
    section out;
    out.name_ = name_;

    if (NULL == root)
        root = &out;

    entry_map::const_iterator eend = entries_.end();
    for (entry_map::const_iterator i = entries_.begin (); i != eend; ++i)
        out.add_entry(i->first, i->second);

    section_map::const_iterator send = sections_.end();
    for (section_map::const_iterator i  = sections_.begin(); i != send; ++i)
    {
        section sub = i->second.clone();
        out.add_section (i->first, i->second.clone(), root);
    }
    return out;
}

inline void indent (int ind, std::ostream& strm) 
{
    for (int i = 0; i < ind; ++i)
        strm << "  ";
}

void section::dump(int ind, std::ostream& strm) const
{
    bool header = false;
    if (0 == ind)
        header = true;

    ++ind;
    if (header)
    {
        if(get_root() == this) {
            strm << "============================\n";
        }
        else {
            strm << "============================[\n"
                 << get_name() << "\n" << "]\n";
        }
    }

    entry_map::const_iterator eend = entries_.end();
    for (entry_map::const_iterator i = entries_.begin(); i != eend; ++i)
    {
        indent (ind, strm);
        strm << "'" << i->first << "' : '" << i->second << "'\n";
    }

    section_map::const_iterator send = sections_.end();
    for (section_map::const_iterator i  = sections_.begin(); i != send; ++i)
    {
        indent (ind, strm);
        strm << "[" << i->first << "]\n";
        (*i).second.dump (ind);
    }

    if (header)
        strm << "============================\n";
}

void section::merge(std::string const& filename)
{
    section tmp(filename);
    merge(tmp);
}

void section::merge(section& second)
{
    // merge entries: keep own entries, and add other entries
    entry_map const& s_entries = second.get_entries();
    entry_map::const_iterator end = s_entries.end();
    for (entry_map::const_iterator i = s_entries.begin(); i != end; ++i)
        entries_[i->first] = i->second;

    // merge subsection known in first section
    section_map::iterator send = sections_.end();
    for (section_map::iterator i = sections_.begin(); i != send; ++i)
    {
        // is there something to merge with?
        if (second.has_section(i->first))
            i->second.merge (second.sections_[i->first]);
    }

    // merge subsection known in second section
    section_map s = second.get_sections();
    section_map::iterator secend = s.end();
    for (section_map::iterator i = s.begin (); i != secend; ++i)
    {
        // if THIS knows the section, we already merged it above
        if (!has_section(i->first))
        {
            // it is not known here, so we can't merge, but have to add it.
            add_section (i->first, i->second, get_root());
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////
bool section::regex_init (void)
{
    int i = 0;
    char const* entry = environ[i];
    while (NULL != entry)
    {
        std::string s (entry);
        std::string::size_type idx = s.find ("=");
        if (idx != std::string::npos)
        {
            std::string key, val;
            if (idx + 1 < s.size())
            {
                key = s.substr(0, idx);
                val = s.substr(idx+1);
            }
            else
            {
                key = s.substr (0, idx);
            }
            section_env_[key] = val;
        }
        entry = environ[++i];
    }
    return true;
}

void section::line_msg(std::string const& msg, std::string const& file, 
    int lnum)
{
    if (lnum > 0)
    {
        boost::throw_exception(hpx::exception(no_success, 
            msg + " " + file + ":" + boost::lexical_cast<std::string>(lnum)));
    }
    boost::throw_exception(hpx::exception(no_success, msg + " " + file));
}

std::string section::expand_entry (std::string value) const
{
    static const boost::regex re(
        // Variable expansion:
        "(?<!\\$)"          //  an even number of dollar signs inhibits variable
/*1*/   "((?:\\$\\$)*)"     //  expansion.

        "\\$"               //  starts with the dollar sign ($);
/*2*/   "(?:\\{|(\\[))"     //  curly brackets ({) for environment variables,
                            //  square brackets ([) for other entries;
/*3*/   "([._[:alnum:]]+)"  //  an alpha-numeric identifier;
        "(?::"              //  an optional default value, starts with a colon
                            //  (:), is used when the variable isn't defined.
/*4*/     "((?:"
/*.*/       "[^\\\\\\{\\[\\]}]" //  Anything goes in here, but brackets ({[]})
/*.*/       "|\\\\[\\{\\[\\]}]" //  and backslashes (\) must be escaped. This
/*.*/                           //  has the nice side effect of allowing
/*.*/                           //  recursion (shhh! :-)
/*.*/     ")*)"
        ")?"
        "(?(2)\\]|\\})"     //  and the corresponding closing bracket (]}).
    );

    boost::smatch match;
    while (boost::regex_search(value, match, re))
    {
        // Get default value
        std::string subst = match.str(4);
        if ("[" == match.str(2))
        {
            std::string setting = get_root()->get_entry(match.str(3), "");
            // Expand with another entry. Empty is considered as undefined, to
            // allow settings to be disabled.
            if (!setting.empty())
            {
                subst = setting;
                // Escape dollar signs to avoid (infinite) recursion
                boost::replace_all(subst, "$", "$$");
            }
        }
        else
        {
            char const * var = ::getenv(match.str(3).c_str());
            if (var)
            {
                subst = var;
                // Escape dollar signs to avoid (infinite) recursion
                boost::replace_all(subst, "$", "$$");
            }
        }

        // Put back prefixing dollar signs, if any.
        subst = match.str(1) + subst;
        value = match.prefix() + subst + match.suffix();
    }

    boost::replace_all(value, "$$", "$");
    return value;
}

}}  // namespace hpx::util

