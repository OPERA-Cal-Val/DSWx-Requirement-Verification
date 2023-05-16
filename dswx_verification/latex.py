from pathlib import Path

MAIN_TEMPLATE_PATH = Path(__file__).parent / 'templates' / 'main_template.tex'
SLIDE_TEMPLATE_PATH = Path(__file__).parent / 'templates' / 'slides.tex'


def get_main_beamer_tex_template() -> str:
    with open(MAIN_TEMPLATE_PATH) as f:
        latex = f.read()
    return latex


def get_slide_tex_template() -> str:
    with open(SLIDE_TEMPLATE_PATH) as f:
        latex = f.read()
    return latex


def render_latex_template(template: str,
                          variables: dict = None,
                          input_paths: dict = None):
    # Strings are passed by value (not reference)
    out = template
    for k, (data_to_render) in enumerate([variables, input_paths]):
        if data_to_render is not None:
            for key, val in data_to_render.items():
                var_key = '\\VAR{' + key + '}'
                if var_key not in template:
                    raise ValueError('key not in Template')
                if k == 1:
                    val = '\\input{' + val + '}'
                out = out.replace(var_key, val)
    return out
