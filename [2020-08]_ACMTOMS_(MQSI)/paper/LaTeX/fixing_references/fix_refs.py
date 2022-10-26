

refj = '''
@journal{{{name},
  authors={{{0}}},
  year={{{1}}},
  title={{{2}}},
  journal={{{3}}},
  volume={{{4}}}
}}
'''.strip()

refb = '''
@book{{{name},
  authors={{{0}}},
  year={{{1}}},
  title={{{2}}},
  publisher={{{3}}}
}}
'''.strip()

reft = '''
@misc{{{name},
  authors={{{0}}},
  year={{{1}}},
  title={{{2}}},
  volume={{{3}}}
}}
'''.strip()


def show_ref(reftype, arguments, name='default'):
    format_string = globals().get(reftype.lstrip("\\"), None)
    if (format_string is not None):
        print(format_string.format(*arguments, name=name))
        print()
    else:
        print(reftype, arguments)
        print()



import re

fname = "old-mqsi20.bib"
with open(fname) as f:
    lines = f.readlines()

comment = None
current_type = ''
current_args = ''
for line in lines:
    # Look for the start of references.
    line = line.strip()
    # Strip out any comments and just print them.
    comment_match = re.search(r'%[^\n]*', line)
    if comment_match:
        comment = comment_match.string[slice(*comment_match.span())]
        line = line.replace(comment, "")
        print(comment)
    # Get the reference type, if that exists.
    reftype = re.search(r'\\ref[\w]*', line)
    # Handle empty lines as the end of a reference.
    if (len(line) == 0):
        if (len(current_type) > 0):
            args = current_args.strip().lstrip("{").rstrip("}").split("}{")
            if (isinstance(comment, str)):
                kwargs = dict(name=comment.lstrip("%").strip())
            else: kwargs = dict()
            show_ref(current_type, args, **kwargs)
            current_type = ''
            current_args = ''
        continue
    elif reftype:
        current_type = reftype.string[slice(*reftype.span())]
        current_args = ''
        line = line[len(current_type)+1:]

    # Process the line.
    current_args += ' '+line
    
