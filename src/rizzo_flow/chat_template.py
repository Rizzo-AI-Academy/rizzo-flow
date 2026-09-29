"""The checkpoint's chat template, rendered the way transformers renders it."""


def compile_template(source: str):
    """A render function for `source`: sandboxed jinja2 with transformers' settings
    (`trim_blocks`, `lstrip_blocks`, `raise_exception`)."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(source).render
