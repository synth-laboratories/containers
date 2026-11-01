"""Strict conversion of the retained Responses trace schema to chat requests."""
import json


def convert(record):
    if record.get('schema') != 'resin.agent-request.v1':
        raise ValueError('unsupported trace schema')
    request = record['request']
    messages = []
    if request.get('instructions'):
        messages.append({'role': 'system', 'content': request['instructions']})
    for item in request.get('input', []):
        typ = item.get('type')
        if typ == 'message':
            role = item['role']
            instruction_role = role if role == 'developer' or (role == 'system' and messages) else None
            if instruction_role:
                # Qwen3.5 only accepts one system message, at the beginning.
                # Keep subsequent instructions in their original history position.
                role = 'user'
            if role not in ('system', 'user', 'assistant'):
                raise ValueError('unsupported message role')
            content = item.get('content', [])
            if isinstance(content, str):
                text = content
            else:
                if any(part.get('type') not in ('input_text', 'output_text', 'text') for part in content):
                    raise ValueError('non-text trace content is unsupported')
                text = ''.join(part['text'] for part in content)
            if instruction_role:
                text = f'[{instruction_role.capitalize()} instruction]\n' + text
            messages.append({'role': role, 'content': text})
        elif typ == 'function_call':
            args = item['arguments']
            json.loads(args)
            call = {'id': item['call_id'], 'type': 'function', 'function': {
                'name': item['name'], 'arguments': args}}
            if messages and messages[-1]['role'] == 'assistant' and messages[-1].get('tool_calls'):
                messages[-1]['tool_calls'].append(call)
            else:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [call]})
        elif typ == 'function_call_output':
            output = item['output']
            if not isinstance(output, str):
                raise ValueError('non-text tool output is unsupported')
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': output})
        else:
            raise ValueError(f'unsupported trace item type {typ!r}')
    tools = []
    for tool in request.get('tools', []):
        if tool.get('type') != 'function':
            raise ValueError('unsupported trace tool type')
        tools.append({'type': 'function', 'function': {k: v for k, v in tool.items() if k != 'type'}})
    if not messages:
        raise ValueError('empty trace request')
    return {'messages': messages, 'tools': tools,
            'recorded_output_tokens': record.get('workload', {}).get('original_output_tokens')}


def read(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(convert(json.loads(line)))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f'{path}:{number}: {exc}') from exc
    if not rows:
        raise ValueError('private panel is empty')
    return rows
