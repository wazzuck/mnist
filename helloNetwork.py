from http.server import BaseHTTPRequestHandler, HTTPServer
import socket


class HelloWorldHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"""
            <html>
                <head>
                    <title>Hello World</title>
                </head>
                <body>
                    <h1>Hello World!</h1>
                    <p>This is a simple Python web server running in Docker.</p>
                </body>
            </html>
        """
        )


def get_local_ip():
    """Get the local IP address that's accessible within the Docker network"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "0.0.0.0"
    finally:
        s.close()
    return IP


def run_server(port=8080):
    server_address = ("0.0.0.0", port)
    local_ip = get_local_ip()
    httpd = HTTPServer(server_address, HelloWorldHandler)
    print(f"Server started on http://{local_ip}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("Server stopped.")


if __name__ == "__main__":
    run_server()
