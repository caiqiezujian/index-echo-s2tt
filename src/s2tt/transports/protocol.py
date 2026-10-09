import json
import struct

from s2tt.types import ProtocolError

HEADER = struct.Struct("<IQ")  # uint32 frame sequence, uint64 source sample-frame offset


def pack_audio(frame_seq, first_sample, pcm16):
    return HEADER.pack(frame_seq, first_sample) + pcm16


def unpack_audio(message):
    if len(message) <= HEADER.size:
        raise ProtocolError("Binary Audio message has no PCM payload")
    seq, offset = HEADER.unpack_from(message)
    return seq, offset, message[HEADER.size:]


def control_message(message):
    try:
        value = json.loads(message)
    except (ValueError, TypeError) as error:
        raise ProtocolError("Control message must be a JSON object") from error
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ProtocolError("Control message requires a string type")
    return value
