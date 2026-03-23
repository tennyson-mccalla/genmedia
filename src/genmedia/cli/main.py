import click


@click.group()
@click.version_option(package_name="genmedia")
def cli():
    """GenMedia — multimodal media generation CLI for Google GenAI."""
    pass


# Import subcommands to register them
from genmedia.cli.image import image  # noqa: E402
from genmedia.cli.edit import edit  # noqa: E402
from genmedia.cli.video import video  # noqa: E402

cli.add_command(image)
cli.add_command(edit)
cli.add_command(video)
