#!/usr/bin/env python3
"""Stream one checkpoint to a GitHub release in resumable, verified parts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

OWNER = "hansonhxmax-eng"
REPO = "RTC-Anything_RobotwinVersion"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"
POLICY = Path("/workspace/RoboTwin/XPolicyLab/policy/Pi_0")
RUN = "rtc_three_suites_pi0_30k"
PART_SIZE = 512 * 1024**2
BLOCK_SIZE = 8 * 1024**2


def credential_token():
    response = subprocess.check_output(
        ["git", "-c", "credential.helper=", "-c", "credential.helper=cache --timeout=1800",
         "-c", "credential.useHttpPath=false", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True,
    )
    return dict(line.split("=", 1) for line in response.splitlines() if "=" in line)["password"]


def publish(step):
    relative = Path("checkpoints") / RUN / str(step)
    if not (POLICY / relative / "params").is_dir():
        raise SystemExit(f"Checkpoint not found: {POLICY / relative}")
    token = credential_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    tag = f"rtc-three-suites-pi0-step-{step}"
    archive = f"rtc_three_suites_pi0_{step}.tar"
    manifest_name = f"{archive}.parts.sha256"

    def api(url, method="GET", data=None):
        payload = None if data is None else json.dumps(data).encode()
        request = Request(url, data=payload, method=method,
                          headers={**headers, "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    releases = api(f"{API}/releases?per_page=100")
    release = next((item for item in releases if item["tag_name"] == tag), None)
    if release and not release["draft"]:
        print(f"Release already published: {release['html_url']}", flush=True)
        return
    if release is None:
        release = api(f"{API}/releases", "POST", {
            "tag_name": tag,
            "target_commitish": "results/rtc-three-suites",
            "name": f"RoboTwin π0 RTC checkpoint {step}",
            "body": (
                "JAX/OpenPI checkpoint from the three-suite RTC simulation run. "
                "Download all numbered parts and the SHA-256 manifest. Verify each part, "
                "concatenate the parts in numeric order, then extract the resulting tar."
            ),
            "draft": True,
        })

    existing = {asset["name"]: asset for asset in api(release["assets_url"] + "?per_page=100")}
    upload_url = release["upload_url"].split("{", 1)[0]
    curl_auth = f'header = "Authorization: Bearer {token}"\n'

    def upload(file, digest):
        found = existing.get(file.name)
        if found and found["state"] == "uploaded":
            if found.get("digest") != f"sha256:{digest}":
                raise RuntimeError(f"Existing release asset has a different checksum: {file.name}")
            print(f"Already uploaded: {file.name}", flush=True)
            return
        if found:
            request = Request(f"{API}/releases/assets/{found['id']}", headers=headers, method="DELETE")
            with urlopen(request, timeout=30):
                pass
        print(f"Uploading {file.name} ({file.stat().st_size:,} bytes)", flush=True)
        result = subprocess.check_output(
            ["curl", "--config", "-", "--fail", "--silent", "--show-error",
             "--connect-timeout", "15", "--max-time", "1800",
             "--header", "Content-Type: application/octet-stream",
             "--data-binary", f"@{file}", f"{upload_url}?name={quote(file.name)}"],
            input=curl_auth, text=True,
        )
        asset = json.loads(result)
        if (asset["state"], asset["size"], asset.get("digest")) != (
            "uploaded", file.stat().st_size, f"sha256:{digest}"
        ):
            raise RuntimeError(f"Release asset did not pass verification: {file.name}")

    manifest = []
    with tempfile.TemporaryDirectory(prefix="rtc-checkpoint-", dir="/workspace") as temporary:
        folder = Path(temporary)
        process = subprocess.Popen(
            ["tar", "--sort=name", "-C", str(POLICY), "-cf", "-", str(relative)],
            stdout=subprocess.PIPE,
        )
        try:
            for index in range(1000):
                part = folder / f"{archive}.part{index:03}"
                digest = hashlib.sha256()
                with part.open("wb") as output:
                    remaining = PART_SIZE
                    while remaining:
                        block = process.stdout.read(min(BLOCK_SIZE, remaining))
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
                if remaining == PART_SIZE:
                    break
                upload(part, digest.hexdigest())
                manifest.append(f"{digest.hexdigest()}  {part.name}\n")
                part.unlink()
            else:
                raise RuntimeError("Checkpoint exceeds the supported part count")
            if process.wait() != 0 or not manifest:
                raise RuntimeError("Checkpoint archive generation failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()

        checksums = folder / manifest_name
        checksums.write_text("".join(manifest), encoding="ascii")
        upload(checksums, hashlib.sha256(checksums.read_bytes()).hexdigest())

    complete = api(f"{API}/releases/{release['id']}", "PATCH", {"draft": False})
    print(f"Checkpoint published: {complete['html_url']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", metavar="STEP", type=int, nargs="*", default=[29999])
    for step in parser.parse_args().steps:
        publish(step)
