"""PyOpenCL compiler cache."""

from __future__ import division

__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

import pyopencl._cl as _cl
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

C_INCLUDE_RE = re.compile(r'^\s*\#\s*include\s+[<"]([^">]+)[">]',
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

                    checksum = new_hash()
                    included_src = src_file.read()
                    checksum.update(included_src)
                    src_file.close()

                    _inner(included_src)

                    result[included_file_name] = (
                            checksum.hexdigest(), os.stat(included_file_name).st_mtime)

                    found = True
                    break # stop searching the include path

            if not found:
                pass

    _inner(src)

    result = list(result.iteritems())
    result.sort()

    return result




def get_file_md5sum(fname):
    checksum = new_hash()
    inf = open(fname)
    checksum.update(inf.read())
    inf.close()
    return checksum.hexdigest()




def check_dependencies(deps):
    for name, date, md5sum in deps:
        try:
            possibly_updated = os.stat(name).st_mtime != date
        except OSError, e:
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




def get_cache_key(device, options, src):
    checksum = new_hash()
    checksum.update(src)
    checksum.update(" ".join(options))
    checksum.update(str(get_device_cache_id(device)))
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
        lock_m = CacheLockManager(cleanup_m, cache_dir)

        mod_cache_dir_m = ModuleCacheDirManager(cleanup_m, module_cache_dir)
        info_path = mod_cache_dir_m.sub("info")
        binary_path = mod_cache_dir_m.sub("binary")
        source_path = mod_cache_dir_m.sub("source.cl")

        # {{{ load info file

        try:
            from cPickle import load

            try:
                info_file = open(info_path)
            except IOError:
                raise _InvalidInfoFile()

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

def _create_built_program_from_source_cached(ctx, src, options, devices, cache_dir):
    include_path = ["."] + [
            option[2:]
            for option in options
            if option.startswith("-I") or option.startswith("/I")]

    if cache_dir is None:
        from os.path import join
        from tempfile import gettempdir
		import getpass
        cache_dir = join(gettempdir(),
                "pyopencl-compiler-cache-v1-uid%s" % getpass.getuser()) 

    # {{{ ensure cache directory exists

    try:
        os.mkdir(cache_dir)
    except OSError, e:
        from errno import EEXIST
        if e.errno != EEXIST:
            raise

    # }}}

    if devices is None:
        devices = ctx.devices

    cache_keys = [get_cache_key(device, options, src) for device in devices]

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
        from warnings import warn
        warn("Build succeeded, but resulted in non-empty logs:\n"+message)
    # {{{ build on the build-needing devices, in one go

    result = None
    already_built = False

    if to_be_built_indices:
        prg = _cl._Program(ctx, src)
        prg.build(options, [devices[i] for i in to_be_built_indices])

        prg_devs = prg.get_info(_cl.program_info.DEVICES)
        prg_bins = prg.get_info(_cl.program_info.BINARIES)
        prg_logs = prg._get_build_logs()

        for i, dest_index in enumerate(to_be_built_indices):
            assert prg_devs[i] == devices[dest_index]
            binaries[dest_index] = prg_bins[i]
            _, logs[dest_index] = prg_logs[i]

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
            lock_m = CacheLockManager(cleanup_m, cache_dir)

            for i in to_be_built_indices:
                cache_key = cache_keys[i]
                device = devices[i]
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

                from cPickle import dump
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




def create_built_program_from_source_cached(ctx, src, options=[], devices=None,
        cache_dir=None):
    try:
        if cache_dir != False:
            prg, already_built = _create_built_program_from_source_cached(
                    ctx, src, options, devices, cache_dir)
        else:
            prg = _cl._Program(ctx, src)
            already_built = False

    except Exception, e:
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
        prg.build(options, devices)

    return prg

# }}}

# vim: foldmethod=marker
