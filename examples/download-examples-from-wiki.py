#! /usr/bin/env python

from __future__ import absolute_import, print_function

import six.moves.xmlrpc_client
destwiki = six.moves.xmlrpc_client.ServerProxy("http://wiki.tiker.net?action=xmlrpc2")

import os
try:
    os.mkdir("wiki-examples")
except OSError:
    pass

print("downloading  wiki examples to wiki-examples/...")
print("fetching page list...")
all_pages = destwiki.getAllPages()


from os.path import exists

for page in all_pages:
    if not page.startswith("PyOpenCL/Examples/"):
        continue

    print(page)
    try:
        content = destwiki.getPage(page)

        import re
        match = re.search(r"\{\{\{\#\!python(.*)\}\}\}", content, re.DOTALL)
        code = match.group(1)

        match = re.search("([^/]+)$", page)
        fname = match.group(1)

        outfname = os.path.join("wiki-examples", fname+".py")
        if exists(outfname):
            print("%s exists, refusing to overwrite." % outfname)
        else:
            outf = open(outfname, "w")
            outf.write(code)
            outf.close()

        for att_name in destwiki.listAttachments(page):
            content = destwiki.getAttachment(page, att_name)

            outfname = os.path.join("wiki-examples", att_name)
            if exists(outfname):
                print("%s exists, refusing to overwrite." % outfname)
            else:
                outf = open(outfname, "w")
                outf.write(str(content))
                outf.close()

    except Exception as e:
        print("Error when processing %s: %s" % (page, e))
        from traceback import print_exc
        print_exc()
