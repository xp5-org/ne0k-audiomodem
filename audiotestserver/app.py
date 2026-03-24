#!/usr/bin/env python3
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Audio Test Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--role",
        choices=["server", "client"],
        required=True,
        help="Run as 'server' (hub + dashboard) or 'client' (agent)",
    )
    parser.add_argument(
        "--id",
        dest="client_id",
        default="client1",
        help="Client identifier (used when --role client)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override default port (server=5000, client=5001)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default 0.0.0.0)",
    )
    args = parser.parse_args()

    if args.role == "server":
        from server_core import run_server
        port = args.port or 8000
        server_id = args.client_id  # --id also names the server's own client entry
        print(f"[server:{server_id}] Starting on http://{args.host}:{port}")
        run_server(host=args.host, port=port, server_id=server_id)

    else:
        from client_core import run_client
        port = args.port or 8001
        print(f"[client:{args.client_id}] Starting agent on http://{args.host}:{port}")
        run_client(client_id=args.client_id, host=args.host, port=port)


if __name__ == "__main__":
    main()