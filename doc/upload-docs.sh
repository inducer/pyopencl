#! /bin/sh

cp build/html/runtime.html build/html/reference.html
rsync --progress --verbose --archive --delete build/html/* doc-upload:doc/pyopencl
