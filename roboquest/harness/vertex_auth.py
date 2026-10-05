"""Private service-account OAuth bridge without changing the simulator runtime.

The system Python supplies PyJWT/cryptography and requests. Its captured output
is consumed only in memory; no credential, JWT or access token enters artifacts.
"""
import json
from pathlib import Path
import stat
import subprocess
import time


_TOKEN_SCRIPT = r'''
import json, sys, time
try:
    import jwt, requests
    with open(sys.argv[1]) as stream:
        credential = json.load(stream)
    endpoint = "https://oauth2.googleapis.com/token"
    if credential.get("type") != "service_account" or credential.get("token_uri") != endpoint:
        raise ValueError("Unsupported credential")
    now = int(time.time())
    assertion = jwt.encode({
        "iss": credential["client_email"],
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": endpoint, "iat": now, "exp": now + 3600,
    }, credential["private_key"], algorithm="RS256")
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(endpoint, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }, timeout=30, allow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError("OAuth exchange rejected")
    token = response.json()
    print(json.dumps({"access_token": token["access_token"], "expires_in": token["expires_in"]}))
except Exception:
    sys.exit(1)
'''


class VertexServiceAccount:
    def __init__(self, credential_file, *, auth_python='/usr/bin/python3'):
        self._path = Path(credential_file).resolve(strict=True)
        info = self._path.stat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError('Vertex credential file must be a regular file with permissions 600')
        try:
            credential = json.loads(self._path.read_text())
            if (credential.get('type') != 'service_account'
                    or credential.get('token_uri') != 'https://oauth2.googleapis.com/token'):
                raise ValueError()
            self.project_id = credential['project_id']
            if not isinstance(self.project_id, str) or not self.project_id:
                raise ValueError()
        except Exception:
            raise ValueError('Invalid Vertex service-account credential') from None
        self._auth_python = str(auth_python)
        self._token = None
        self._expires_at = 0.

    def token(self):
        if self._token is not None and time.monotonic() < self._expires_at:
            return self._token
        try:
            # Starting expiry before the child request is conservative.
            started = time.monotonic()
            process = subprocess.run(
                [self._auth_python, '-I', '-c', _TOKEN_SCRIPT, str(self._path)],
                capture_output=True, text=True, timeout=45, check=False)
            if process.returncode:
                raise RuntimeError()
            response = json.loads(process.stdout)
            token, expires = response['access_token'], float(response['expires_in'])
            if not isinstance(token, str) or not token or not 120 < expires <= 3600:
                raise ValueError()
            self._token, self._expires_at = token, started + expires - 120
            return self._token
        except Exception:
            # Never expose captured subprocess stdout/stderr or raw OAuth errors.
            raise RuntimeError('Vertex OAuth token acquisition failed') from None
