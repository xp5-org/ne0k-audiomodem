# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""Nested function assigns a name owned by an enclosing function, and READS it
before the first assignment -> Python treats it as a fresh local and the read
raises UnboundLocalError.  A plain temp that is assigned before use is fine, so
ordering (not mere name reuse) is the test."""
import ast, sys

def _params(fn):
    a = fn.args
    out = {p.arg for p in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg: out.add(a.vararg.arg)
    if a.kwarg:  out.add(a.kwarg.arg)
    return out

def _own_nodes(fn):
    """Walk fn's body but do not descend into nested function/class scopes."""
    stack, out = list(fn.body), []
    while stack:
        n = stack.pop()
        out.append(n)
        # Comprehensions and lambdas have their OWN scope in Python 3, so
        # their targets are not locals of `fn` -- descending into them reports
        # every `for x in ...` temp as a shadowed enclosing variable.
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                          ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                          ast.GeneratorExp)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return out

def _bound(fn):
    out = set()
    for n in _own_nodes(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out

def _declared(fn):
    out = set()
    for n in _own_nodes(fn):
        if isinstance(n, (ast.Nonlocal, ast.Global)):
            out.update(n.names)
    return out

bad = []
for path in sys.argv[1:]:
    tree = ast.parse(open(path).read(), path)
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for outer in funcs:
        owned = _bound(outer) | _params(outer)
        inners = [n for n in _own_nodes(outer)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for inner in inners:
            risky = (_bound(inner) & owned) - _declared(inner) - _params(inner)
            for nm in sorted(risky):
                pos = [(n.lineno, n.col_offset, type(n.ctx).__name__)
                       for n in _own_nodes(inner)
                       if isinstance(n, ast.Name) and n.id == nm]
                pos.sort()
                if pos and pos[0][2] == "Load":       # read precedes any store
                    bad.append((path, pos[0][0], outer.name, inner.name, nm))

for p, ln, o, i, nm in sorted(bad):
    print(f"  {p}:{ln}  {o}() -> {i}(): reads enclosing '{nm}' before assigning "
          f"it, no nonlocal -> UnboundLocalError")
print(f"{len(bad)} finding(s)")
sys.exit(1 if bad else 0)
