"""A series of functions to test if a string is a valid qName

$ ID:   $


"""

import string
from unicodedata import category
from set_importer import Set

LETTER_CATEGORIES = Set(["Ll", "Lu", "Lo", "Lt", "Nl"])
NCNAME_CATEGORIES = LETTER_CATEGORIES.union(Set(["Mc", "Me", "Mn", "Lm", "Nd"]))

NCNameChar, NCNameStartChar, NameStartChar, NameChar, \
            Letter, Digit, CombiningChar, Extender, BaseChar, Ideographic = \
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9

def isXMLChar10(character, char_class):
    """Is this a valid <char_class> character
    
    char_class is one of NCNameChar, NCNameStartChar, NameStartChar, NameChar, 
            Letter, Digit, CombiningChar, Extender, BaseChar, Ideographic

    usual use is isXMLChar(character, isXML.NCNameChar)
    """
    num = ord(character)
    character = unicode(character)
    if char_class == Letter:
        return isXMLChar10(character, BaseChar) or \
               isXMLChar10(character, Ideographic)
### The following appears to simply be too accepting
###        return category(character) in LETTER_CATEGORIES
    elif char_class == NCNameStartChar:
        return (character == '_') or \
               isXMLChar10(character, Letter)
    elif char_class == NCNameChar:
        return (character == '-') or \
               (character == '.') or \
               isXMLChar10(character, NCNameStartChar) or \
               isXMLChar10(character, Digit) or \
               isXMLChar10(character, CombiningChar) or \
               isXMLChar10(character, Extender)
### The following appears to simply be too accepting
###               (character == '_') or \
###               category(character) in NCNAME_CATEGORIES
    elif char_class == NameStartChar:
        return character == ':' or isXMLChar10(character, NCNameStartChar)
    elif char_class == NameChar:
        return character == ':' or isXMLChar10(character, NCNameChar)
    elif char_class == BaseChar:
        return (num >= 0x0041 and num <= 0x005A) or \
               (num >= 0x0061 and num <= 0x007A) or \
               (num >= 0x00C0 and num <= 0x00D6) or \
               (num >= 0x00D8 and num <= 0x00F6) or \
               (num >= 0x00F8 and num <= 0x00FF) or \
               (num >= 0x0100 and num <= 0x0131) or \
               (num >= 0x0134 and num <= 0x013E) or \
               (num >= 0x0141 and num <= 0x0148) or \
               (num >= 0x014A and num <= 0x017E) or \
               (num >= 0x0180 and num <= 0x01C3) or \
               (num >= 0x01CD and num <= 0x01F0) or \
               (num >= 0x01F4 and num <= 0x01F5) or \
               (num >= 0x01FA and num <= 0x0217) or \
               (num >= 0x0250 and num <= 0x02A8) or \
               (num >= 0x02BB and num <= 0x02C1) or \
               (num == 0x0386) or \
               (num >= 0x0388 and num <= 0x038A) or \
               (num == 0x038C) or \
               (num >= 0x038E and num <= 0x03A1) or \
               (num >= 0x03A3 and num <= 0x03CE) or \
               (num >= 0x03D0 and num <= 0x03D6) or \
               (num == 0x03DA) or \
               (num == 0x03DC) or \
               (num == 0x03DE) or \
               (num == 0x03E0) or \
               (num >= 0x03E2 and num <= 0x03F3) or \
               (num >= 0x0401 and num <= 0x040C) or \
               (num >= 0x040E and num <= 0x044F) or \
               (num >= 0x0451 and num <= 0x045C) or \
               (num >= 0x045E and num <= 0x0481) or \
               (num >= 0x0490 and num <= 0x04C4) or \
               (num >= 0x04C7 and num <= 0x04C8) or \
               (num >= 0x04CB and num <= 0x04CC) or \
               (num >= 0x04D0 and num <= 0x04EB) or \
               (num >= 0x04EE and num <= 0x04F5) or \
               (num >= 0x04F8 and num <= 0x04F9) or \
               (num >= 0x0531 and num <= 0x0556) or \
               (num == 0x0559) or \
               (num >= 0x0561 and num <= 0x0586) or \
               (num >= 0x05D0 and num <= 0x05EA) or \
               (num >= 0x05F0 and num <= 0x05F2) or \
               (num >= 0x0621 and num <= 0x063A) or \
               (num >= 0x0641 and num <= 0x064A) or \
               (num >= 0x0671 and num <= 0x06B7) or \
               (num >= 0x06BA and num <= 0x06BE) or \
               (num >= 0x06C0 and num <= 0x06CE) or \
               (num >= 0x06D0 and num <= 0x06D3) or \
               (num == 0x06D5) or \
               (num >= 0x06E5 and num <= 0x06E6) or \
               (num >= 0x0905 and num <= 0x0939) or \
               (num == 0x093D) or \
               (num >= 0x0958 and num <= 0x0961) or \
               (num >= 0x0985 and num <= 0x098C) or \
               (num >= 0x098F and num <= 0x0990) or \
               (num >= 0x0993 and num <= 0x09A8) or \
               (num >= 0x09AA and num <= 0x09B0) or \
               (num == 0x09B2) or \
               (num >= 0x09B6 and num <= 0x09B9) or \
               (num >= 0x09DC and num <= 0x09DD) or \
               (num >= 0x09DF and num <= 0x09E1) or \
               (num >= 0x09F0 and num <= 0x09F1) or \
               (num >= 0x0A05 and num <= 0x0A0A) or \
               (num >= 0x0A0F and num <= 0x0A10) or \
               (num >= 0x0A13 and num <= 0x0A28) or \
               (num >= 0x0A2A and num <= 0x0A30) or \
               (num >= 0x0A32 and num <= 0x0A33) or \
               (num >= 0x0A35 and num <= 0x0A36) or \
               (num >= 0x0A38 and num <= 0x0A39) or \
               (num >= 0x0A59 and num <= 0x0A5C) or \
               (num == 0x0A5E) or \
               (num >= 0x0A72 and num <= 0x0A74) or \
               (num >= 0x0A85 and num <= 0x0A8B) or \
               (num == 0x0A8D) or \
               (num >= 0x0A8F and num <= 0x0A91) or \
               (num >= 0x0A93 and num <= 0x0AA8) or \
               (num >= 0x0AAA and num <= 0x0AB0) or \
               (num >= 0x0AB2 and num <= 0x0AB3) or \
               (num >= 0x0AB5 and num <= 0x0AB9) or \
               (num == 0x0ABD) or \
               (num == 0x0AE0) or \
               (num >= 0x0B05 and num <= 0x0B0C) or \
               (num >= 0x0B0F and num <= 0x0B10) or \
               (num >= 0x0B13 and num <= 0x0B28) or \
               (num >= 0x0B2A and num <= 0x0B30) or \
               (num >= 0x0B32 and num <= 0x0B33) or \
               (num >= 0x0B36 and num <= 0x0B39) or \
               (num == 0x0B3D) or \
               (num >= 0x0B5C and num <= 0x0B5D) or \
               (num >= 0x0B5F and num <= 0x0B61) or \
               (num >= 0x0B85 and num <= 0x0B8A) or \
               (num >= 0x0B8E and num <= 0x0B90) or \
               (num >= 0x0B92 and num <= 0x0B95) or \
               (num >= 0x0B99 and num <= 0x0B9A) or \
               (num == 0x0B9C) or \
               (num >= 0x0B9E and num <= 0x0B9F) or \
               (num >= 0x0BA3 and num <= 0x0BA4) or \
               (num >= 0x0BA8 and num <= 0x0BAA) or \
               (num >= 0x0BAE and num <= 0x0BB5) or \
               (num >= 0x0BB7 and num <= 0x0BB9) or \
               (num >= 0x0C05 and num <= 0x0C0C) or \
               (num >= 0x0C0E and num <= 0x0C10) or \
               (num >= 0x0C12 and num <= 0x0C28) or \
               (num >= 0x0C2A and num <= 0x0C33) or \
               (num >= 0x0C35 and num <= 0x0C39) or \
               (num >= 0x0C60 and num <= 0x0C61) or \
               (num >= 0x0C85 and num <= 0x0C8C) or \
               (num >= 0x0C8E and num <= 0x0C90) or \
               (num >= 0x0C92 and num <= 0x0CA8) or \
               (num >= 0x0CAA and num <= 0x0CB3) or \
               (num >= 0x0CB5 and num <= 0x0CB9) or \
               (num == 0x0CDE) or \
               (num >= 0x0CE0 and num <= 0x0CE1) or \
               (num >= 0x0D05 and num <= 0x0D0C) or \
               (num >= 0x0D0E and num <= 0x0D10) or \
               (num >= 0x0D12 and num <= 0x0D28) or \
               (num >= 0x0D2A and num <= 0x0D39) or \
               (num >= 0x0D60 and num <= 0x0D61) or \
               (num >= 0x0E01 and num <= 0x0E2E) or \
               (num == 0x0E30) or \
               (num >= 0x0E32 and num <= 0x0E33) or \
               (num >= 0x0E40 and num <= 0x0E45) or \
               (num >= 0x0E81 and num <= 0x0E82) or \
               (num == 0x0E84) or \
               (num >= 0x0E87 and num <= 0x0E88) or \
               (num == 0x0E8A) or \
               (num == 0x0E8D) or \
               (num >= 0x0E94 and num <= 0x0E97) or \
               (num >= 0x0E99 and num <= 0x0E9F) or \
               (num >= 0x0EA1 and num <= 0x0EA3) or \
               (num == 0x0EA5) or \
               (num == 0x0EA7) or \
               (num >= 0x0EAA and num <= 0x0EAB) or \
               (num >= 0x0EAD and num <= 0x0EAE) or \
               (num == 0x0EB0) or \
               (num >= 0x0EB2 and num <= 0x0EB3) or \
               (num == 0x0EBD) or \
               (num >= 0x0EC0 and num <= 0x0EC4) or \
               (num >= 0x0F40 and num <= 0x0F47) or \
               (num >= 0x0F49 and num <= 0x0F69) or \
               (num >= 0x10A0 and num <= 0x10C5) or \
               (num >= 0x10D0 and num <= 0x10F6) or \
               (num == 0x1100) or \
               (num >= 0x1102 and num <= 0x1103) or \
               (num >= 0x1105 and num <= 0x1107) or \
               (num == 0x1109) or \
               (num >= 0x110B and num <= 0x110C) or \
               (num >= 0x110E and num <= 0x1112) or \
               (num == 0x113C) or \
               (num == 0x113E) or \
               (num == 0x1140) or \
               (num == 0x114C) or \
               (num == 0x114E) or \
               (num == 0x1150) or \
               (num >= 0x1154 and num <= 0x1155) or \
               (num == 0x1159) or \
               (num >= 0x115F and num <= 0x1161) or \
               (num == 0x1163) or \
               (num == 0x1165) or \
               (num == 0x1167) or \
               (num == 0x1169) or \
               (num >= 0x116D and num <= 0x116E) or \
               (num >= 0x1172 and num <= 0x1173) or \
               (num == 0x1175) or \
               (num == 0x119E) or \
               (num == 0x11A8) or \
               (num == 0x11AB) or \
               (num >= 0x11AE and num <= 0x11AF) or \
               (num >= 0x11B7 and num <= 0x11B8) or \
               (num == 0x11BA) or \
               (num >= 0x11BC and num <= 0x11C2) or \
               (num == 0x11EB) or \
               (num == 0x11F0) or \
               (num == 0x11F9) or \
               (num >= 0x1E00 and num <= 0x1E9B) or \
               (num >= 0x1EA0 and num <= 0x1EF9) or \
               (num >= 0x1F00 and num <= 0x1F15) or \
               (num >= 0x1F18 and num <= 0x1F1D) or \
               (num >= 0x1F20 and num <= 0x1F45) or \
               (num >= 0x1F48 and num <= 0x1F4D) or \
               (num >= 0x1F50 and num <= 0x1F57) or \
               (num == 0x1F59) or \
               (num == 0x1F5B) or \
               (num == 0x1F5D) or \
               (num >= 0x1F5F and num <= 0x1F7D) or \
               (num >= 0x1F80 and num <= 0x1FB4) or \
               (num >= 0x1FB6 and num <= 0x1FBC) or \
               (num == 0x1FBE) or \
               (num >= 0x1FC2 and num <= 0x1FC4) or \
               (num >= 0x1FC6 and num <= 0x1FCC) or \
               (num >= 0x1FD0 and num <= 0x1FD3) or \
               (num >= 0x1FD6 and num <= 0x1FDB) or \
               (num >= 0x1FE0 and num <= 0x1FEC) or \
               (num >= 0x1FF2 and num <= 0x1FF4) or \
               (num >= 0x1FF6 and num <= 0x1FFC) or \
               (num == 0x2126) or \
               (num >= 0x212A and num <= 0x212B) or \
               (num == 0x212E) or \
               (num >= 0x2180 and num <= 0x2182) or \
               (num >= 0x3041 and num <= 0x3094) or \
               (num >= 0x30A1 and num <= 0x30FA) or \
               (num >= 0x3105 and num <= 0x312C) or \
               (num >= 0xAC00 and num <= 0xD7A3)
    elif char_class == Ideographic:
        return (num >= 0x4E00 and num <= 0x9FA5) or \
               (num == 0x3007) or \
               (num >= 0x3021 and num <= 0x3029)
    elif char_class == Digit:
        return (num >= 0x0030 and num <= 0x0039) or \
               (num >= 0x0660 and num <= 0x0669) or \
               (num >= 0x06F0 and num <= 0x06F9) or \
               (num >= 0x0966 and num <= 0x096F) or \
               (num >= 0x09E6 and num <= 0x09EF) or \
               (num >= 0x0A66 and num <= 0x0A6F) or \
               (num >= 0x0AE6 and num <= 0x0AEF) or \
               (num >= 0x0B66 and num <= 0x0B6F) or \
               (num >= 0x0BE7 and num <= 0x0BEF) or \
               (num >= 0x0C66 and num <= 0x0C6F) or \
               (num >= 0x0CE6 and num <= 0x0CEF) or \
               (num >= 0x0D66 and num <= 0x0D6F) or \
               (num >= 0x0E50 and num <= 0x0E59) or \
               (num >= 0x0ED0 and num <= 0x0ED9) or \
               (num >= 0x0F20 and num <= 0x0F29)
    elif char_class == CombiningChar:
        return (num >= 0x0300 and num <= 0x0345) or \
               (num >= 0x0360 and num <= 0x0361) or \
               (num >= 0x0483 and num <= 0x0486) or \
               (num >= 0x0591 and num <= 0x05A1) or \
               (num >= 0x05A3 and num <= 0x05B9) or \
               (num >= 0x05BB and num <= 0x05BD) or \
               (num == 0x05BF) or \
               (num >= 0x05C1 and num <= 0x05C2) or \
               (num == 0x05C4) or \
               (num >= 0x064B and num <= 0x0652) or \
               (num == 0x0670) or \
               (num >= 0x06D6 and num <= 0x06DC) or \
               (num >= 0x06DD and num <= 0x06DF) or \
               (num >= 0x06E0 and num <= 0x06E4) or \
               (num >= 0x06E7 and num <= 0x06E8) or \
               (num >= 0x06EA and num <= 0x06ED) or \
               (num >= 0x0901 and num <= 0x0903) or \
               (num == 0x093C) or \
               (num >= 0x093E and num <= 0x094C) or \
               (num == 0x094D) or \
               (num >= 0x0951 and num <= 0x0954) or \
               (num >= 0x0962 and num <= 0x0963) or \
               (num >= 0x0981 and num <= 0x0983) or \
               (num == 0x09BC) or \
               (num == 0x09BE) or \
               (num == 0x09BF) or \
               (num >= 0x09C0 and num <= 0x09C4) or \
               (num >= 0x09C7 and num <= 0x09C8) or \
               (num >= 0x09CB and num <= 0x09CD) or \
               (num == 0x09D7) or \
               (num >= 0x09E2 and num <= 0x09E3) or \
               (num == 0x0A02) or \
               (num == 0x0A3C) or \
               (num == 0x0A3E) or \
               (num == 0x0A3F) or \
               (num >= 0x0A40 and num <= 0x0A42) or \
               (num >= 0x0A47 and num <= 0x0A48) or \
               (num >= 0x0A4B and num <= 0x0A4D) or \
               (num >= 0x0A70 and num <= 0x0A71) or \
               (num >= 0x0A81 and num <= 0x0A83) or \
               (num == 0x0ABC) or \
               (num >= 0x0ABE and num <= 0x0AC5) or \
               (num >= 0x0AC7 and num <= 0x0AC9) or \
               (num >= 0x0ACB and num <= 0x0ACD) or \
               (num >= 0x0B01 and num <= 0x0B03) or \
               (num == 0x0B3C) or \
               (num >= 0x0B3E and num <= 0x0B43) or \
               (num >= 0x0B47 and num <= 0x0B48) or \
               (num >= 0x0B4B and num <= 0x0B4D) or \
               (num >= 0x0B56 and num <= 0x0B57) or \
               (num >= 0x0B82 and num <= 0x0B83) or \
               (num >= 0x0BBE and num <= 0x0BC2) or \
               (num >= 0x0BC6 and num <= 0x0BC8) or \
               (num >= 0x0BCA and num <= 0x0BCD) or \
               (num == 0x0BD7) or \
               (num >= 0x0C01 and num <= 0x0C03) or \
               (num >= 0x0C3E and num <= 0x0C44) or \
               (num >= 0x0C46 and num <= 0x0C48) or \
               (num >= 0x0C4A and num <= 0x0C4D) or \
               (num >= 0x0C55 and num <= 0x0C56) or \
               (num >= 0x0C82 and num <= 0x0C83) or \
               (num >= 0x0CBE and num <= 0x0CC4) or \
               (num >= 0x0CC6 and num <= 0x0CC8) or \
               (num >= 0x0CCA and num <= 0x0CCD) or \
               (num >= 0x0CD5 and num <= 0x0CD6) or \
               (num >= 0x0D02 and num <= 0x0D03) or \
               (num >= 0x0D3E and num <= 0x0D43) or \
               (num >= 0x0D46 and num <= 0x0D48) or \
               (num >= 0x0D4A and num <= 0x0D4D) or \
               (num == 0x0D57) or \
               (num == 0x0E31) or \
               (num >= 0x0E34 and num <= 0x0E3A) or \
               (num >= 0x0E47 and num <= 0x0E4E) or \
               (num == 0x0EB1) or \
               (num >= 0x0EB4 and num <= 0x0EB9) or \
               (num >= 0x0EBB and num <= 0x0EBC) or \
               (num >= 0x0EC8 and num <= 0x0ECD) or \
               (num >= 0x0F18 and num <= 0x0F19) or \
               (num == 0x0F35) or \
               (num == 0x0F37) or \
               (num == 0x0F39) or \
               (num == 0x0F3E) or \
               (num == 0x0F3F) or \
               (num >= 0x0F71 and num <= 0x0F84) or \
               (num >= 0x0F86 and num <= 0x0F8B) or \
               (num >= 0x0F90 and num <= 0x0F95) or \
               (num == 0x0F97) or \
               (num >= 0x0F99 and num <= 0x0FAD) or \
               (num >= 0x0FB1 and num <= 0x0FB7) or \
               (num == 0x0FB9) or \
               (num >= 0x20D0 and num <= 0x20DC) or \
               (num == 0x20E1) or \
               (num >= 0x302A and num <= 0x302F) or \
               (num == 0x3099) or \
               (num == 0x309A)
    elif char_class == Extender:
        return (num == 0x00B7) or \
               (num == 0x02D0) or \
               (num == 0x02D1) or \
               (num == 0x0387) or \
               (num == 0x0640) or \
               (num == 0x0E46) or \
               (num == 0x0EC6) or \
               (num == 0x3005) or \
               (num >= 0x3031 and num <= 0x3035) or \
               (num >= 0x309D and num <= 0x309E) or \
               (num >= 0x30FC and num <= 0x30FE)
    else:
        raise NotImplementedError
        

def isXMLChar11(character, char_class):
    """Is this a valid <char_class> character
    
    char_class is one of NCNameChar, NCNameStartChar, NameStartChar, NameChar, 
            Letter, Digit, CombiningChar, Extender, BaseChar, Ideographic

    usual use is isXMLChar(character, isXML.NCNameChar)
    """
    num = ord(character)
    if char_class == NCNameStartChar:
        return (character in string.lowercase) or \
               (character in string.uppercase) or \
               (character == '_') or \
               (num >= 0xC0 and num <= 0xD6) or \
               (num >= 0xD8 and num <= 0xF6) or \
               (num >= 0xF8 and num <= 0x2FF) or \
               (num >= 0x370 and num <= 0x37D) or \
               (num >= 0x37F and num <= 0x1FFF) or \
               (num >= 0x200C and num <= 0x200D) or \
               (num >= 0x2070 and num <= 0x218F) or \
               (num >= 0x2C00 and num <= 0x2FEF) or \
               (num >= 0x3001 and num <= 0xD7FF) or \
               (num >= 0xF900 and num <= 0xFDCF) or \
               (num >= 0xFDF0 and num <= 0xFFFD) or \
               (num >= 0x10000 and num <= 0xEFFFF)  
    elif char_class == NCNameChar:
        return (character in string.digits) or \
               (character == '-') or \
               (character == '.') or \
               (num == 0xB7) or \
               (num >= 0x300 and num <= 0x36F) or \
               (num >= 0x203F and num <= 0x2040) or \
               isXMLChar11(character, NCNameStartChar)
    elif char_class == NameStartChar:
        return character == ':' or isXMLChar11(character, NCNameStartChar)
    elif char_class == NameChar:
        return character == ':' or isXMLChar11(character, NCNameChar)
    else:
        raise NotImplementedError

def isXMLChar(character, char_class):
    """Is this a valid <char_class> character
    
    char_class is one of NCNameChar, NCNameStartChar, NameStartChar, NameChar, 
            Letter, Digit, CombiningChar, Extender, BaseChar, Ideographic

    usual use is isXMLChar(character, isXML.NCNameChar)
    """ 
    if XMLVersion == '1.1':
        return isXMLChar11(character, char_class)
    else:
        return isXMLChar10(character, char_class)


def isNCName(string):
    """Is this string a valid NCName

    """
    if not isXMLChar(string[0], NCNameStartChar):
        return False
    for a in string[1:]:
        if not isXMLChar(a, NCNameChar):
            return False
    return True

def isName(string):
    """Is this string a valid NCName

    """
    if not isXMLChar(string[0], NameStartChar):
        return False
    for a in string[1:]:
        if not isXMLChar(a, NameChar):
            return False
    return True


XMLVersion = '1.0'

def setXMLVersion(ver):
    global XMLVersion
    XMLVersion = ver

def getXMLVersion():
    return XMLVersion
