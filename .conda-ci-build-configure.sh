case $(uname) in
  MINGW*|MSYS*) python ./configure.py --cl-inc-dir="$CONDA_PREFIX/Library/include" \
	  --cl-lib-dir="$CONDA_PREFIX/Library/lib";;
  *) python ./configure.py --cl-inc-dir="$CONDA_PREFIX/include" \
	  --cl-lib-dir="$CONDA_PREFIX/lib";;
esac
