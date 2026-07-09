# Makes `util` a regular package so the vendored/custom modules are importable as
# `util.MATCC_dataset`, `util.DropExtremeLabel`, etc. (needed so qlib can resolve the
# class __module__ when pickling the fitted handler cache).
