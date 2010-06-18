#! /bin/sh

cp build/html/runtime.rst build/html/reference.rst
rsync --progress --verbose --archive --delete build/html/* buster:doc/pyopencl
