"""
A webserver for SPARQL

"""
__version__ = '$Id: webserver.py,v 1.9 2005/08/09 20:55:16 syosi Exp $'

import BaseHTTPServer, urllib
from cgi import parse_qs
from sys import exc_info

def sparql_handler(s):
    return s

default_string = '''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>

  <meta content="text/html; charset=ISO-8859-1" http-equiv="content-type" />
  <title>Cwm SPARQL Server</title>


</head>



<body>

The Cwm SPARQL server at %s<br />

<br />

Enter a query here:<br />

<form method="get" action="" name="sparql query"><textarea cols="80" rows="5" name="query">PREFIX log: <http://www.w3.org/2000/10/swap/log#>
SELECT * {log:includes ?x ?y}</textarea><br />

  <button style="height: 2em; width: 5em;" name="Submit">Submit</button></form>

</body>
</html>'''

##file_open = file
##m_types = {
##    'html': 'text/html',
##    'css': 'text/css',
##    'js': 'text/javascript',
##    'gif': 'image/gif',
##    'jpeg': 'image/jpeg'
##    }


class SPARQL_request_handler(BaseHTTPServer.BaseHTTPRequestHandler):
    server_version = "CWM/" + __version__[1:-1]
    query_file = '/'
    default = default_string % query_file
    def do_GET(self):
        try:
            file, query = self.path.split('?', 1)
        except ValueError:
            file, query = self.path, ''
##        if file[:13] == '/presentation':
##            try:
##                dot = file.rfind('.')
##                if dot>0:
##                    c_type = m_types.get(file[dot+1:], 'text/plain')
##                else:
##                    c_type = 'text/plain'
##                s = file_open('/home/syosi/server_present/'+file[14:], 'r').read()
##                self.send_response(200)
##                self.send_header("Content-type", c_type)
##                self.send_header("Content-Length", str(len(s)))
##                self.end_headers()
##                self.wfile.write(s)
##                return
##            except:
##                pass
        if file != self.query_file:
            self.send_error(404, "File not found")
            return
        args = parse_qs(query)

        print args

        query = args.get('query', [''])[0]
        if not query:
            self.send_response(200)
            self.send_header("Content-type", 'text/html')
            self.send_header("Content-Length", str(len(self.default)))
            self.end_headers()
            self.wfile.write(self.default)
        else:
            try:
                retVal, ctype= sparql_handler(query)
            except:
                self.send_response(400)
                resp = str(exc_info()[1])
                print 'error is', resp
                self.send_header("Content-type", 'text/plain')
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                retVal = retVal.encode('utf_8')
                print ctype
                self.send_response(200)
                self.send_header("Content-type", ctype)
                self.send_header("Content-Length", str(len(retVal)))
                self.end_headers()
                self.wfile.write(retVal)
    do_POST = do_GET



def run(server_class=BaseHTTPServer.HTTPServer,
        handler_class=SPARQL_request_handler):
    server_address = ('', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()





if __name__ == '__main__':
    run()
