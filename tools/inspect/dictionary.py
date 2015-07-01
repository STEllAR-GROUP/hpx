#  Header making script  ------------------------------------------//

#  Copyright (c) 2015 Brandon Cordes
#  Distributed under the Boost Software License, Version 1.0.
#  (See accompanying file LICENSE_1_0.txt or copy at
#  http://www.boost.org/LICENSE_1_0.txt)

out = open('dictionary.hpp', 'w')
out.writelines("//  extra_whitespace_check header  -------------------------")
out.writelines("-----------------//\n\n//  Copyright (c) 2015 Brandon Cordes")
out.writelines("\n//  Distributed under the Boost Software License")
out.writelines(", Version 1.0.\n//  (See accompanying file LICENSE_1_0.txt")
out.writelines(" or copy at\n//  http://www.boost.org/LICENSE_1_0.txt)\n")
out.writelines("\n#ifndef DICTIONARY_HPP\n#define DICTIONARY_HPP\n\n")
out.writelines("#include <vector>\n\nchar const* const dictionary[] = {\n")
with open('dictionary.txt') as f:
    lines = [line.rstrip('\n') for line in open('dictionary.txt')]
for i in range (len(lines)):
    out.writelines("   \"" + lines[i] + "\",\n")
out.writelines("};\n\n#endif\n")