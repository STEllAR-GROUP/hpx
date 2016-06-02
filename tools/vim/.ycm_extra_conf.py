# Copyright (c) 2016 Marcin Copik
# Copyright (c) 2016 Parsa Amini
#
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import os
import re

compilation_database_dir = "@CMAKE_BINARY_DIR@"

def FlagsForFile(filename, **kwargs):

    a = ''
    #Ignore lines not containing CXX command
    with open('{0}/compile_commands.json'.format(compilation_database_dir), 'r') as f:
        a = ''.join([l.rstrip('\n') for l in f.readlines() if 'command' in l])

    final_flags = []
    
    final_flags.extend(set(re.findall('-std\S+', a)))
    #Allow any number of whitespace between include flag and directory
    final_flags.extend(set(re.findall('-I\s*\S+', a)))
    final_flags.extend(set(re.findall('-include\s*\S+', a)))
    final_flags.extend(set(re.findall('-isystem\s*\S+', a)))
    final_flags.extend(set(re.findall('-W\S+', a)))
    final_flags.extend(set(re.findall('-w\S+', a)))
 
    return {
        'flags': final_flags,
        'do_cache': True
    }

