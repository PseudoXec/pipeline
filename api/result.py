from enum import Enum


class SendResult(str, Enum):
    SENT = "sent"        # written to the WebSocket
    FAILED = "failed"    # socket down / write failed - the buffer keeps the record and retries after reconnect
    SKIPPED = "skipped"  # nothing to send (e.g. vehicle without a plate) - not an error
    REJECTED = "rejected"  # this record can never be sent (e.g. not JSON-serializable) - dropped, not retried
