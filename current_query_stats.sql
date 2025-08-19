SELECT pid,
       now() - query_start AS duration,
       state,
       wait_event_type,
       wait_event,
       query
FROM pg_stat_activity
WHERE state = 'active'
ORDER BY duration DESC
LIMIT 100;
