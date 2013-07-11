#! /bin/sh

rsync --progress --verbose --archive --delete _build/html/* doc-upload:doc/pyopencl
