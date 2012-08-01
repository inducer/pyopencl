" Vim highlighting for PyOpenCL
" -----------------------------
"
" (C) Andreas Kloeckner 2011, MIT license
"
" Uses parts of mako.vim by Armin Ronacher.
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

" {{{ mako

syn region clmakoLine start="^\s*%" skip="\\$" end="$"
syn region clmakoVariable start=#\${# end=#}# contains=@pythonTop
syn region clmakoBlock start=#<%!# end=#%># keepend contains=@pythonTop

syn match clmakoAttributeKey containedin=clmakoTag contained "[a-zA-Z_][a-zA-Z0-9_]*="
syn region clmakoAttributeValue containedin=clmakoTag contained start=/"/ skip=/\\"/ end=/"/
syn region clmakoAttributeValue containedin=clmakoTag contained start=/'/ skip=/\\'/ end=/'/

syn region clmakoTag start="</\?%\(def\|call\|page\|include\|namespace\|inherit\|self:[_[:alnum:]]\+\)\>" end="/\?>"

" The C highlighter's paren error detection screws up highlighting of 
" Mako variables in C parens--turn it off.

syn clear cParen
syn clear cParenError
if !exists("c_no_bracket_error")
  syn clear cBracket
endif

syn cluster clmakoCode contains=clmakoLine,clmakoVariable,clmakoBlock,clmakoTag

hi link clmakoLine Preproc
hi link clmakoVariable Preproc
hi link clmakoBlock Preproc
hi link clmakoTag Define
hi link clmakoAttributeKey String
hi link clmakoAttributeValue String

" }}}

syn region pythonCLString
      \ start=+[uU]\=\z('''\|"""\)//CL\(:[a-zA-Z_0-9]\+\)\?//+ end="\z1" keepend
      \ contains=@clCode,@clmakoCode

syn region pythonCLRawString
      \ start=+[uU]\=[rR]\z('''\|"""\)//CL\(:[a-zA-Z_0-9]\+\)\?//+ end="\z1" keepend
      \ contains=@clCode,@clmakoCode

" Uncomment if you still want the code highlighted as a string.
" hi link pythonCLString String
" hi link pythonCLRawString String

syntax sync fromstart

let b:current_syntax = "pyopencl"

" vim: foldmethod=marker
