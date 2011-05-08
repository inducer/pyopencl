" Vim highlighting for PyOpenCL
" -----------------------------
"
" (C) Andreas Kloeckner 2011, MIT license
"
" Installation:
" Just drop this file into ~/.vim/syntax/pyopencl.vim
"
" Then do 
" :set filetype=pyopencl
" and use 
" """//CL// ...code..."""
" for OpenCL code included in your Python file.
"
" You may also include a line
" vim: filetype=pyopencl.python
" at the end of your file to set the file type automatically.
"
" Optional: Install opencl.vim from
" http://www.vim.org/scripts/script.php?script_id=3157

runtime! syntax/python.vim

unlet b:current_syntax
try
  syntax include @clCode syntax/opencl.vim
catch
  syntax include @clCode syntax/c.vim
endtry

unlet b:current_syntax
syn include @pythonTop syntax/python.vim

syn region makoLine start="^\s*%" skip="\\$" end="$"
syn region makoVariable start=#\${# end=#}# contains=@pythonTop
syn clear cParen
syn clear cParenError

syn cluster makoCode contains=makoLine,makoVariable

syn region pythonCLString
      \ start=+[uU]\=\z('''\|"""\)//CL//+ end="\z1" keepend
      \ contains=@clCode,@makoCode

hi link makoLine Special
hi link makoVariable Special
hi link pythonCLString String

let b:current_syntax = "pyopencl"
