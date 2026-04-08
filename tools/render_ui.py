"""Render UI Tool — generative UI components rendered inline in chat.

Emits a structured declarative payload that the frontend translates into
interactive visual components (dashboards, forms, approval cards, tables,
activity feeds, etc.).  Two variants are registered:

  render_ui     — named composition templates with typed data bindings
  render_custom — freeform block arrays for one-off layouts

The agent should prefer render_ui when one of the known compositions fits;
fall back to render_custom when the desired layout doesn't map to any template.
"""

import json
import logging

logger = logging.getLogger(__name__)

RENDER_UI_SCHEMA = {
    'name': 'render_ui',
    'description': (
        'Render an interactive UI component inline in the chat. '
        'Use a named composition template (render_ui) for common layouts, '
        'or render_custom for freeform block arrays. '
        'The component appears visually in the conversation immediately.\n\n'
        'Always call render_ui when showing dashboards, approval requests, '
        'structured data tables, multi-step forms, or any content that would '
        'benefit from a richer visual presentation than plain text.\n\n'
        'Use the optional componentId to enable in-place updates for multi-step '
        'components (wizards, multi-step forms). If componentId is provided and '
        'a component with that ID is already rendered, the frontend updates it '
        'in-place rather than creating a new component.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'composition': {
                'type': 'string',
                'description': (
                    'The composition template to render. '
                    'Known compositions: kpi_dashboard, approval_card, triage_table, '
                    'form_wizard, env_vars_form, comparison_view, activity_feed, email_reply. '
                    'Custom compositions may be defined by skills.'
                ),
            },
            'data': {
                'type': 'object',
                'description': (
                    'Template data used to populate the composition. '
                    'The keys depend on the chosen composition. '
                    'Examples: '
                    'kpi_dashboard -> {metrics, chart, table}; '
                    'approval_card -> {title, action, rationale, risks, options}; '
                    'triage_table -> {items, actions}; '
                    'form_wizard -> {steps: [{label, fields: [{name, label, type, hint}]}], currentStep: 0}; '
                    'env_vars_form -> {service, description, fields}; '
                    'comparison_view -> {options}; '
                    'activity_feed -> {entries, actions}; '
                    'email_reply -> {subject, from, from_email, original_message, draft_reply, reply_actions}.'
                ),
            },
            'title': {
                'type': 'string',
                'description': 'Optional title shown above the rendered component.',
            },
            'componentId': {
                'type': 'string',
                'description': (
                    'Optional stable identifier for multi-step components. '
                    'If provided and a component with this ID is already rendered, '
                    'the frontend will update it in-place.'
                ),
            },
        },
        'required': ['composition', 'data'],
    },
}

RENDER_CUSTOM_SCHEMA = {
    'name': 'render_custom',
    'description': (
        'Render a freeform UI component from an explicit block array. '
        'Use when no named composition fits the desired layout. '
        'For standard layouts prefer render_ui instead.'
    ),
    'parameters': {
        'type': 'object',
        'properties': {
            'blocks': {
                'type': 'array',
                'description': (
                    'Array of block objects. '
                    'Supported block types: text, alert, badge, table, form, '
                    'actions, card, columns, entries, metrics, chart, stepper, '
                    'avatar, divider.'
                ),
                'items': {'type': 'object'},
            },
            'title': {
                'type': 'string',
                'description': 'Optional title shown above the component.',
            },
        },
        'required': ['blocks'],
    },
}


def _render_ui_handler(args: dict, **kw) -> str:
    payload = json.dumps(args, ensure_ascii=False)
    return f'[RENDER_UI]{payload}[/RENDER_UI]'


def _render_custom_handler(args: dict, **kw) -> str:
    payload = json.dumps(args, ensure_ascii=False)
    return f'[RENDER_UI]{payload}[/RENDER_UI]'


from tools.registry import registry  # noqa: E402

registry.register(
    name='render_ui',
    toolset='render_ui',
    schema=RENDER_UI_SCHEMA,
    handler=_render_ui_handler,
    emoji='🎨',
)

registry.register(
    name='render_custom',
    toolset='render_ui',
    schema=RENDER_CUSTOM_SCHEMA,
    handler=_render_custom_handler,
    emoji='🧩',
)
