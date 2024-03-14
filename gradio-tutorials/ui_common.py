import gradio as gr

refresh_symbol = '\U0001f504'  # 🔄


class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form

gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent


class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"
    
def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = gr.Button(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button