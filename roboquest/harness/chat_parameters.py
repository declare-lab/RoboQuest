"""Chat Completions request preparation, applied before generation capture and retries."""

import json

import httpx


class ChatParametersTransport(httpx.BaseTransport):
    """Make each Chat Completions request acceptable to a strict OpenAI-compatible server.

    The harness keeps its conversation in Chat Completions form with one private message flag, ``cache_anchor``
    (prompt-cache breakpoints for the Anthropic wire). This transport removes it and, when the model configures an
    output limit, adds ``max_tokens``. Wrap ``RetryingGenerationTransport`` with it so that each counted attempt and
    its wire capture hold exactly what was sent. It owns no connection, authentication or retry loop.
    """

    def __init__(self, inner, *, max_output_tokens=None):
        if not isinstance(inner, httpx.BaseTransport):
            raise ValueError('Chat parameters transport requires an inner transport')
        if max_output_tokens is not None and (isinstance(max_output_tokens, bool)
                                              or not isinstance(max_output_tokens, int) or max_output_tokens < 1):
            raise ValueError('max_output_tokens must be a positive integer or None')
        self.inner = inner
        self._max_output_tokens = max_output_tokens

    def handle_request(self, request):
        if request.method != 'POST' or not request.url.path.endswith('/chat/completions'):
            raise ValueError('Chat parameters require a Chat Completions POST request')
        try:
            payload = json.loads(request.read())
        except (ValueError, UnicodeError):
            raise ValueError('Chat Completions body must be a JSON object') from None
        if not isinstance(payload, dict) or not isinstance(payload.get('messages'), list):
            raise ValueError('Chat Completions body must be a JSON object with messages')
        for message in payload['messages']:
            if isinstance(message, dict):
                message.pop('cache_anchor', None)
        if self._max_output_tokens is not None:
            payload.setdefault('max_tokens', self._max_output_tokens)
        content = json.dumps(payload, allow_nan=False).encode()
        headers = httpx.Headers(request.headers)
        headers.pop('content-length', None)
        headers.pop('transfer-encoding', None)
        outgoing = httpx.Request(request.method, request.url, headers=headers,
                                 content=content, extensions=dict(request.extensions))
        return self.inner.handle_request(outgoing)

    def close(self):
        self.inner.close()
