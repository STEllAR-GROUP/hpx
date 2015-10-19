//  Hyperlink Function  ------------------------------------------//

//  Copyright (c) 2015 Brandon Cordes
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef FUNCTION_HYPER_HPP
#define FUNCTION_HYPER_HPP

#include "boost/filesystem/path.hpp"
#include <hpx/config/defines.hpp>
#include <string>

using boost::filesystem::path;
//When you have a specific line and the line is the location of the link
inline std::string linelink(const path & full_path, std::string linenumb)
{
    std::string commit = HPX_HAVE_GIT_COMMIT;
    std::string total, location = full_path.string();
    total += "†a href = \"https://github.com/STEllAR-GROUP/hpx/blob/" + commit + "/" + location + "#L" + linenumb + "\"‡";
    total = total + linenumb;
    total = total + "†/a‡";
    return total;
}
//When you have a specific line, but a word is the location of the link
inline std::string wordlink(const path & full_path, std::string linenumb, std::string word)
{
    std::string commit = HPX_HAVE_GIT_COMMIT;
    std::string total, location = full_path.string();
    total += "†a href = \"https://github.com/STEllAR-GROUP/hpx/blob/" + commit + "/" + location + "#L" + linenumb + "\"‡";
    total = total + word;
    total = total + "†/a‡";
    return total;
}
//When you don't have a specific line
inline std::string loclink(const path & full_path, std::string word)
{
    std::string commit = HPX_HAVE_GIT_COMMIT;
    std::string total, location = full_path.string();
    total += "†a href = \"https://github.com/STEllAR-GROUP/hpx/blob/" + commit + "/" + location + "\"‡";
    total = total + word;
    total = total + "†/a‡";
    return total;
}
#endif