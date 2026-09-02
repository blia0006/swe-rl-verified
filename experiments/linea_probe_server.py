#!/usr/bin/env python3
"""线 A 通路探针服务：在 TKE Pod 内监听 8000，供沙箱经 NodePort 访问。

只依赖标准库 —— Pod 内不装任何额外包。
"""
import http.server
import socketserver

TOKEN = "SWE_RL_LINEA_OK"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(('{"probe":"%s"}' % TOKEN).encode())

    def log_message(self, *args):
        pass  # 静音，避免污染日志


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", 8000), Handler) as srv:
        srv.serve_forever()
