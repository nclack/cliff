# 2026-01-01

Zig `comptime` is pretty good.

I've been able to use it to generalize the blade product. You can't really
capture the type requirements in the type signature. A generic type has to be
anytype. But you can check the invariants and generate a compiler error.

It's very much a duck-typing approach, so you have to check the invariants early
if you don't want to run into compiler errors more deeply into the code.

And you can dynamically generate structs. I used that to autogenerate the
basis blades for a particular space.

I don't think you can manipulate the ast, but you can fill out the metadata
for the type.
