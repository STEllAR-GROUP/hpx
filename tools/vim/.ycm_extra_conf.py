# Copyright (c) 2016 Marcin Copik
# Copyright (c) 2016 Parsa Amini

import os
import re
 
def FlagsForFile(filename, **kwargs):

    a = []
    with open('compile_commands.json', 'r') as f:
        a = ''.join([l.rstrip('\n') for l in f.readlines() if 'command' in l])

    final_flags = []
    
    final_flags.extend(set(re.findall('-std\S+', a)))
    final_flags.extend(set(re.findall('-I\S+', a)))
    final_flags.extend(set(re.findall('-W\S+', a)))
    final_flags.extend(set(re.findall('-w\S+', a)))
 
    return {
        'flags': final_flags,
        'do_cache': True
    }
