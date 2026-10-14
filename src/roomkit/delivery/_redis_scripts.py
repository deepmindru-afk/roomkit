"""Atomic Redis delivery transitions, fenced by pending-entry ownership."""

from __future__ import annotations

# Check ownership before any write. Repeating a transition after losing its
# response is harmless: the old entry no longer belongs to this consumer.
# With a delay (ARGV[6], milliseconds), the replacement waits in the delayed
# set KEYS[2], scored by the server's clock, until PROMOTE queues it again.
TRANSITION = """
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 or pending[1][2] ~= ARGV[3] then return 0 end
if ARGV[4] ~= '' then
    if ARGV[6] ~= '' then
        local now = redis.call('TIME')
        local due = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
        redis.call('ZADD', KEYS[2], due + tonumber(ARGV[6]), ARGV[4])
    elseif tonumber(ARGV[5]) > 0 then
        redis.call('XADD', KEYS[2], 'MAXLEN', '~', ARGV[5], '*', 'data', ARGV[4])
    else
        redis.call('XADD', KEYS[2], '*', 'data', ARGV[4])
    end
end
redis.call('XACK', KEYS[1], ARGV[1], ARGV[2])
redis.call('XDEL', KEYS[1], ARGV[2])
return 1
"""

# Queue again the retries whose delay has passed: KEYS[1] the delayed set,
# KEYS[2] the pending stream, ARGV[1] how many at most. Atomic, so two workers
# never queue one retry twice.
PROMOTE = """
local now = redis.call('TIME')
local ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local due = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ms, 'LIMIT', 0, tonumber(ARGV[1]))
for _, member in ipairs(due) do
    redis.call('XADD', KEYS[2], '*', 'data', member)
    redis.call('ZREM', KEYS[1], member)
end
return #due
"""

# A slow, live worker renews only entries it still owns. XCLAIM without this
# check could steal an entry back from its replacement after a disconnection.
RENEW = """
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[2], ARGV[2], 1)
if #pending == 0 or pending[1][2] ~= ARGV[3] then return 0 end
redis.call('XCLAIM', KEYS[1], ARGV[1], ARGV[3], 0, ARGV[2], 'JUSTID')
return 1
"""
