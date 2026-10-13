from enum import Enum


class SendResult(str, Enum):
    SENT = "sent"          # the dashboard accepted it - row can be deleted / marked synced
    FAILED = "failed"      # try again later
    REJECTED = "rejected"  # the API answered with a permanent 4xx - retrying will never help
