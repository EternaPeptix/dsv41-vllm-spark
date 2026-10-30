#!/usr/bin/env python3
"""lanproxy.py — expose the DSv4.1 API (vLLM on 127.0.0.1:8000, rank 0) on spark-one's LAN and Tailscale addresses.

Plain TCP forwarder (stdlib asyncio), so streaming/SSE and every endpoint pass through unchanged and
vLLM itself is not restarted or reconfigured. Binds the current IPv4 of each of LAN_IFACES (default "wlP9s9 tailscale0") on
LISTEN_PORT (default 8000); exits when any of those addresses changes so systemd restarts it on the new one.
User request 2026-10-26: "going forward, let's expose the API to the LAN".
"""
import asyncio, os, socket, subprocess, sys

IFACES = os.environ.get("LAN_IFACES", "wlP9s9 tailscale0").split()
PORT = int(os.environ.get("LISTEN_PORT", "8000"))
UPSTREAM = ("127.0.0.1", int(os.environ.get("UPSTREAM_PORT", "8000")))


def lan_ip(iface):
    out = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", iface], capture_output=True, text=True).stdout
    for tok in out.split():
        if "/" in tok and tok[0].isdigit():
            return tok.split("/")[0]
    return None


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(creader, cwriter):
    try:
        ureader, uwriter = await asyncio.open_connection(*UPSTREAM)
    except OSError:
        cwriter.close()          # vLLM restarting: refuse quickly, the client retries
        return
    for w in (cwriter, uwriter):
        s = w.get_extra_info("socket")
        if s is not None:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    await asyncio.gather(pipe(creader, uwriter), pipe(ureader, cwriter))


async def main():
    ips = {i: lan_ip(i) for i in IFACES}
    up = {i: ip for i, ip in ips.items() if ip}
    if not up:
        print(f"lanproxy: no IPv4 on any of {IFACES}", file=sys.stderr)
        sys.exit(1)
    servers = []
    for iface, ip in up.items():
        servers.append(await asyncio.start_server(handle, ip, PORT, reuse_address=True, backlog=1024))
        print(f"lanproxy: {iface} {ip}:{PORT} -> {UPSTREAM[0]}:{UPSTREAM[1]}", flush=True)
    while True:
        await asyncio.sleep(30)
        if {i: lan_ip(i) for i in IFACES} != ips:   # an address appeared, vanished or moved: restart
            print(f"lanproxy: addresses changed from {ips}; exiting for restart", flush=True)
            sys.exit(3)

asyncio.run(main())
