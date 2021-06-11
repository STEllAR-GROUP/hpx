'''
Copyright (c) 2020 ETH Zurich

SPDX-License-Identifier: BSL-1.0
Distributed under the Boost Software License, Version 1.0. (See accompanying
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
'''
import contextlib
import os
import pathlib
from lxml import etree as et

from pyutils import log

_CSS = '''
    /* general style */
    body {
        font-family: sans-serif;
        display: flex;
        flex-wrap: wrap;
    }
    section {
        flex: 0 0 auto;
        align-items: center;
        box-sizing: border-box;
        padding: 20px;
    }
    h1 {
        width: 100%;
        padding: 20px;
    }
    /* table style */
    table {
        padding-bottom: 5em;
        border-collapse: collapse;
        table-layout:fixed;
        max-width: calc(100vw - 60px);
    }
    th {
        text-align: left;
        border-bottom: 1px solid black;
        padding: 0.5em;
        word-wrap:break-word;
    }
    td {
        padding: 0.5em;
        word-wrap:break-word;
    }
    /* styles for good/bad/unknown entries in table */
    .good {
        color: #81b113;
        background: #dfff79;
        font-weight: bold;
    }
    .bad {
        color: #c23424;
        background: #ffd0ac;
        font-weight: bold;
    }
    .unknown {
        color: #1f65c2;
        background: #d5ffff;
        font-weight: bold;
    }
    /* image-grid style */
    .grid-section {
        width: 100%;
    }
    .grid-container {
        width: 100%;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    }
    img {
        width: 100%;
        max-width: 50em;
    }

'''


class Report:
    def __init__(self, data_dir, title):
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir.mkdir()
        self.data_counter = 0

        self.html = et.Element('html')
        self._init_head()
        self.body = et.SubElement(self.html, 'body')
        header = et.SubElement(self.body, 'h1')
        header.text = title

    def _init_head(self):
        head = et.SubElement(self.html, 'head')
        et.SubElement(head, 'meta', charset='utf-8')
        et.SubElement(head,
                      'meta',
                      name='viewport',
                      content='width=device-width, initial-scale=1.0')
        et.SubElement(head,
                      'link',
                      rel='stylesheet',
                      href=str(self._write_css()))

    def _write_css(self):
        path, rel_path = self.get_data_path('.css')
        with path.open('w') as css_file:
            css_file.write(_CSS)
        log.debug(f'Successfully written CSS to {path}')
        return rel_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.write()
        return False

    def write(self):
        et.ElementTree(self.html).write(str(os.path.join(self.data_dir,
            'index.html')),
                                        encoding='utf-8',
                                        method='html')
        log.info(f'Successfully written HTML report to {self.data_dir}')

    def get_data_path(self, suffix=''):
        filename = f'{self.data_counter:03}{suffix}'
        self.data_counter += 1
        return self.data_dir / filename, filename

    @contextlib.contextmanager
    def _section(self, title):
        section = et.SubElement(self.body, 'section')
        if title:
            header = et.SubElement(section, 'h2')
            header.text = title
        yield section

    @contextlib.contextmanager
    def table(self, title=None):
        with self._section(title) as section:
            yield _Table(section)

    @contextlib.contextmanager
    def image_grid(self, title=None):
        with self._section(title) as section:
            section.set('class', 'grid-section')
            yield _Grid(section, self.get_data_path)


class _Table:
    def __init__(self, parent):
        self.html = et.SubElement(parent, 'table')
        self.first = True

    @contextlib.contextmanager
    def row(self):
        yield _TableRow(self.html, self.first)
        self.first = False


class _TableRow:
    def __init__(self, parent, header):
        self.html = et.SubElement(parent, 'tr')
        self.header = header

    def cell(self, text):
        elem = et.SubElement(self.html, 'th' if self.header else 'td')
        elem.text = text
        return elem

    def fill(self, *texts):
        return [self.cell(text) for text in texts]


class _Grid:
    def __init__(self, parent, get_data_path):
        self.html = et.SubElement(parent, 'div', {'class': 'grid-container'})
        self.get_data_path = get_data_path

    def image(self):
        path, rel_path = self.get_data_path('.png')
        et.SubElement(self.html, 'img', {
            'class': 'grid-item',
            'src': str(rel_path)
        })
        return path
