import logging
import datetime as dt
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import logging_setup


def test_intercept_handler_binds_stdlib_extra_fields():
    handler = logging_setup._InterceptHandler()

    with patch.object(logging_setup, 'logger') as mock_logger:
        mock_bound = MagicMock()
        mock_opt = MagicMock()
        mock_opt.bind.return_value = mock_bound
        mock_logger.opt.return_value = mock_opt
        mock_logger.level.return_value = SimpleNamespace(name='INFO')

        record = logging.LogRecord(
            name='test.logger',
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg='hello',
            args=(),
            exc_info=None,
        )
        record.otel_trace_id = 'trace-123'
        record.message_id = 'msg-123'

        handler.emit(record)

        mock_opt.bind.assert_called_once()
        bound_kwargs = mock_opt.bind.call_args.kwargs
        assert bound_kwargs['otel_trace_id'] == 'trace-123'
        assert bound_kwargs['message_id'] == 'msg-123'
        mock_bound.log.assert_called_once_with('INFO', 'hello')


def test_json_sink_promotes_trace_fields_from_extra():
    writes = []

    class _Stdout:
        def write(self, text):
            writes.append(text)

        def flush(self):
            return None

    fake_message = SimpleNamespace(
        record={
            'time': dt.datetime.now(dt.timezone.utc),
            'level': SimpleNamespace(name='INFO'),
            'message': 'hello',
            'name': 'mod',
            'function': 'fn',
            'line': 1,
            'extra': {
                'otel_trace_id': 'trace-abc',
                'message_id': 'msg-abc',
            },
            'exception': None,
        }
    )

    with patch.object(logging_setup, 'sys', SimpleNamespace(stdout=_Stdout())):
        logging_setup._json_sink(fake_message)

    assert writes
    payload = json.loads(writes[0])
    assert payload['otel_trace_id'] == 'trace-abc'
    assert payload['message_id'] == 'msg-abc'
