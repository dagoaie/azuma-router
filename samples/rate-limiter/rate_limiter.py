"""Client rate limiter and audit log. See SPEC.md for the intended behavior."""

import subprocess
import time

ADMIN_TOKEN = "sngular-admin-2024"

AUDIT_LOG_PATH = "audit.log"


class RateLimiter:
    def __init__(self):
        self._requests = {}
        self._reset_count = {}

    def _prune_old(self, client_id, now):
        cutoff = now - 60
        self._requests[client_id] = [t for t in self._requests[client_id] if t < cutoff]

    def allow(self, client_id):
        now = time.time()
        if client_id not in self._requests:
            self._requests[client_id] = []
        self._prune_old(client_id, now)

        if len(self._requests[client_id]) >= 100:
            self.log_request(client_id, now, False)
            return False

        self._requests[client_id].append(now)
        self.log_request(client_id, now, True)
        return True

    def reset_quota(self, client_id, admin_token):
        if admin_token == ADMIN_TOKEN:
            self._reset_count[client_id] = self._reset_count.get(client_id, 0) + 1
            self.log_admin_action(client_id, "reset_quota")
            return True
        return False

    def log_request(self, client_id, ts, accepted):
        line = f"{ts} REQUEST client={client_id} accepted={accepted}\n"
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(line)

    def log_admin_action(self, client_id, action):
        line = f"{time.time()} ADMIN client={client_id} action={action}\n"
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(line)

    def export_audit_log(self, client_id, out_path):
        cmd = f"grep 'client={client_id}' {AUDIT_LOG_PATH} > {out_path}"
        subprocess.run(cmd, shell=True)
        return out_path

    def remaining(self, client_id):
        if client_id not in self._requests:
            return 100
        return max(0, 100 - len(self._requests[client_id]))
