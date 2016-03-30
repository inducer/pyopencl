"""PyOpenCL compiler cache."""

from __future__ import division
from __future__ import absolute_import
import six
from six.moves import zip

__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

#import pyopencl._cl as _cl
import pyopencl.cffi_cl as _cl
import re
import sys
import os
from pytools import Record

try:
    import hashlib
    new_hash = hashlib.md5
except ImportError:
    # for Python << 2.5
    import md5
    new_hash = md5.new


def _erase_dir(dir):
    from os import listdir, unlink, rmdir
    from os.path import join
    for name in listdir(dir):
        unlink(join(dir, name))
    rmdir(dir)


def update_checksum(checksum, obj):
    if isinstance(obj, six.text_type):
        checksum.update(obj.encode("utf8"))
    else:
        checksum.update(obj)


# {{{ cleanup

class CleanupBase(object):
    pass


class CleanupManager(CleanupBase):
    def __init__(self):
        self.cleanups = []

    def register(self, c):
        self.cleanups.insert(0, c)

    def clean_up(self):
        for c in self.cleanups:
            c.clean_up()

    def error_clean_up(self):
        for c in self.cleanups:
            c.error_clean_up()


class CacheLockManager(CleanupBase):
    def __init__(self, cleanup_m, cache_dir):
        if cache_dir is not None:
            self.lock_file = os.path.join(cache_dir, "lock")

            attempts = 0
            while True:
                try:
                    self.fd = os.open(self.lock_file,
                            os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                    break
                except OSError:
                    pass

                from time import sleep
                sleep(1)

                attempts += 1

                if attempts > 10:
                    from warnings import warn
                    warn("could not obtain cache lock--delete '%s' if necessary"
                            % self.lock_file)

            cleanup_m.register(self)

    def clean_up(self):
        import os
        os.close(self.fd)
        os.unlink(self.lock_file)

    def error_clean_up(self):
        pass


class ModuleCacheDirManager(CleanupBase):
    def __init__(self, cleanup_m, path):
        from os import mkdir

        self.path = path
        try:
            mkdir(self.path)
            cleanup_m.register(self)
            self.existed = False
        except OSError:
            self.existed = True

    def sub(self, n):
        from os.path import join
        return join(self.path, n)

    def reset(self):
        import os
        _erase_dir(self.path)
        os.mkdir(self.path)

    def clean_up(self):
        pass

    def error_clean_up(self):
        _erase_dir(self.path)

# }}}


# {{{ #include dependency handling

C_INCLUDE_RE = re.compile(r'^\s*\#\s*include\s+[<"](.+)[">]\s*$',
        re.MULTILINE)


def get_dependencies(src, include_path):
    result = {}

    from os.path import realpath, join

    def _inner(src):
        for match in C_INCLUDE_RE.finditer(src):
            included = match.group(1)

            found = False
            for ipath in include_path:
                included_file_name = realpath(join(ipath, included))

                if included_file_name not in result:
                    try:
                        src_file = open(included_file_name, "rt")
                    except IOError:
                        continue

                    try:
                        included_src = src_file.read()
                    finally:
                        src_file.close()

                    # jrevent infinite recursion if some header file appears to
                    # include itself
                    result[included_file_name] = None

                    checksum = new_hash()
                    update_checksum(checksum, included_src)
                    _inner(included_src)

                    result[included_file_name] = (
                            os.stat(included_file_name).st_mtime,
                            checksum.hexdigest(),
                            )

                    found = True
                    break  # stop searching the include path

            if not found:
                pass

    _inner(src)

    result = list((name,) + vals for name, vals in six.iteritems(result))
    result.sort()

    return result


def get_file_md5sum(fname):
    checksum = new_hash()
    inf = open(fname)
    try:
        contents = inf.read()
    finally:
        inf.close()
    update_checksum(checksum, contents)
    return checksum.hexdigest()


def check_dependencies(deps):
    for name, date, md5sum in deps:
        try:
            possibly_updated = os.stat(name).st_mtime != date
        except OSError:
            return False
        else:
            if possibly_updated and md5sum != get_file_md5sum(name):
                return False

    return True

# }}}


# {{{ key generation

def get_device_cache_id(device):
    from pyopencl.version import VERSION
    platform = device.platform
    return (VERSION,
            platform.vendor, platform.name, platform.version,
            device.vendor, device.name, device.version, device.driver_version)


def get_cache_key(device, options_bytes, src):
    checksum = new_hash()
    update_checksum(checksum, src)
    update_checksum(checksum, options_bytes)
    update_checksum(checksum, str(get_device_cache_id(device)))
    return checksum.hexdigest()

# }}}


def retrieve_from_cache(cache_dir, cache_key):
    class _InvalidInfoFile(RuntimeError):
        pass

    from os.path import join, isdir
    module_cache_dir = join(cache_dir, cache_key)
    if not isdir(module_cache_dir):
        return None

    cleanup_m = CleanupManager()
    try:
        try:
            CacheLockManager(cleanup_m, cache_dir)

            mod_cache_dir_m = ModuleCacheDirManager(cleanup_m, module_cache_dir)
            info_path = mod_cache_dir_m.sub("info")
            binary_path = mod_cache_dir_m.sub("binary")

            # {{{ load info file

            try:
                from six.moves.cPickle import load

                try:
                    info_file = open(info_path, "rb")
                except IOError:
                    raise _InvalidInfoFile()

                try:
                    try:
                        info = load(info_file)
                    except EOFError:
                        raise _InvalidInfoFile()
                finally:
                    info_file.close()

            except _InvalidInfoFile:
                mod_cache_dir_m.reset()
                from warnings import warn
                warn("PyOpenCL encountered an invalid info file for cache key %s"
                        % cache_key)
                return None

            # }}}

            # {{{ load binary

            binary_file = open(binary_path, "rb")
            try:
                binary = binary_file.read()
            finally:
                binary_file.close()

            # }}}

            if check_dependencies(info.dependencies):
                return binary, info.log
            else:
                mod_cache_dir_m.reset()

        except:
            cleanup_m.error_clean_up()
            raise
    finally:
        cleanup_m.clean_up()


# {{{ top-level driver

class _SourceInfo(Record):
    pass


def _create_built_program_from_source_cached(ctx, src, options_bytes,
        devices, cache_dir, include_path):
    from os.path import join

    if cache_dir is None:
        import appdirs
        cache_dir = join(appdirs.user_cache_dir("pyopencl", "pyopencl"),
                "pyopencl-compiler-cache-v2-py%s" % (
                    ".".join(str(i) for i in sys.version_info),))

    # {{{ ensure cache directory exists

    try:
        os.makedirs(cache_dir)
    except OSError as e:
        from errno import EEXIST
        if e.errno != EEXIST:
            raise

    # }}}

    if devices is None:
        devices = ctx.devices

    cache_keys = [get_cache_key(device, options_bytes, src) for device in devices]

    binaries = []
    to_be_built_indices = []
    logs = []
    for i, (device, cache_key) in enumerate(zip(devices, cache_keys)):
        cache_result = retrieve_from_cache(cache_dir, cache_key)

        if cache_result is None:
            to_be_built_indices.append(i)
            binaries.append(None)
            logs.append(None)
        else:
            binary, log = cache_result
            binaries.append(binary)
            logs.append(log)

    message = (75*"="+"\n").join(
            "Build on %s succeeded, but said:\n\n%s" % (dev, log)
            for dev, log in zip(devices, logs)
            if log is not None and log.strip())

    if message:
        from pyopencl import compiler_output
        compiler_output(
                "Built kernel retrieved from cache. Original from-source "
                "build had warnings:\n"+message)

    # {{{ build on the build-needing devices, in one go

    result = None
    already_built = False

    if to_be_built_indices:
        # defeat implementation caches:
        from uuid import uuid4
        src = src + "\n\n__constant int pyopencl_defeat_cache_%s = 0;" % (
                uuid4().hex)

        prg = _cl._Program(ctx, src)
        prg.build(options_bytes, [devices[i] for i in to_be_built_indices])

        prg_devs = prg.get_info(_cl.program_info.DEVICES)
        prg_bins = prg.get_info(_cl.program_info.BINARIES)
        prg_logs = prg._get_build_logs()

        for dest_index in to_be_built_indices:
            dev = devices[dest_index]
            src_index = prg_devs.index(dev)
            binaries[dest_index] = prg_bins[src_index]
            _, logs[dest_index] = prg_logs[src_index]

        if len(to_be_built_indices) == len(devices):
            # Important special case: if code for all devices was built,
            # then we may simply use the program that we just built as the
            # final result.

            result = prg
            already_built = True

    if result is None:
        result = _cl._Program(ctx, devices, binaries)

    # }}}

    # {{{ save binaries to cache

    if to_be_built_indices:
        cleanup_m = CleanupManager()
        try:
            try:
                CacheLockManager(cleanup_m, cache_dir)

                for i in to_be_built_indices:
                    cache_key = cache_keys[i]
                    binary = binaries[i]

                    mod_cache_dir_m = ModuleCacheDirManager(cleanup_m,
                            join(cache_dir, cache_key))
                    info_path = mod_cache_dir_m.sub("info")
                    binary_path = mod_cache_dir_m.sub("binary")
                    source_path = mod_cache_dir_m.sub("source.cl")

                    outf = open(source_path, "wt")
                    outf.write(src)
                    outf.close()

                    outf = open(binary_path, "wb")
                    outf.write(binary)
                    outf.close()

                    from six.moves.cPickle import dump
                    info_file = open(info_path, "wb")
                    dump(_SourceInfo(
                        dependencies=get_dependencies(src, include_path),
                        log=logs[i]), info_file)
                    info_file.close()

            except:
                cleanup_m.error_clean_up()
                raise
        finally:
            cleanup_m.clean_up()

    # }}}

    return result, already_built


def create_built_program_from_source_cached(ctx, src, options_bytes, devices=None,
        cache_dir=None, include_path=None):
    try:
        if cache_dir is not False:
            prg, already_built = _create_built_program_from_source_cached(
                    ctx, src, options_bytes, devices, cache_dir,
                    include_path=include_path)
        else:
            prg = _cl._Program(ctx, src)
            already_built = False

    except Exception as e:
        raise
        from pyopencl import Error
        if (isinstance(e, Error)
                and e.code == _cl.status_code.BUILD_PROGRAM_FAILURE):
            # no need to try again
            raise

        from warnings import warn
        from traceback import format_exc
        warn("PyOpenCL compiler caching failed with an exception:\n"
                "[begin exception]\n%s[end exception]"
                % format_exc())

        prg = _cl._Program(ctx, src)
        already_built = False

    if not already_built:
        prg.build(options_bytes, devices)

    return prg

# }}}

# vim: foldmethod=marker
